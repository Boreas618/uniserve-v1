//! Starting the deployment's engine process and connecting to it.
//!
//! The leader daemon starts each engine process as a child of its own node's
//! supervision, through the process interface like any process of the
//! deployment: [`EngineControl`] connects to the daemon's process socket and
//! asks it to spawn `uniserve engine`, which connects back to the daemon's
//! request socket and says hello. The daemon answers with the engine's
//! specification and waits until the engine reports that it serves; the
//! engine's process lives for as long as its control connection does.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use proto::process::{self, command, downstream, event, hello, reply, report, request, upstream};
use proto::request::wire;
use proto::{Connection, Listener};

use crate::engine::client::{EngineClient, EngineState, StatsSink};

/// The worker group a daemon supervises an engine process under.
const ENGINE_GROUP: &str = "engine";

/// How long an engine process takes to connect back to the request socket
/// once the daemon started it.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// How long the daemon takes to answer the control connection.
const DAEMON_TIMEOUT: Duration = Duration::from_secs(90);

/// How often a wait rechecks the engine process while it starts.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Starts the engine processes of one deployment.
pub struct EngineControl {
    /// The request socket engine processes connect back to.
    requests: Listener,
    request_address: String,
    socket: PathBuf,
    /// The daemon's process socket.
    processes: String,
    /// This node's host identity, where the engine process runs.
    host: String,
    /// The deployment the engines serve.
    deployment: String,
    next_engine: AtomicU64,
    /// Engine handoffs can connect in any order. Reading and assigning
    /// their greetings is serialized; each client keeps its buffered reader.
    incoming: std::sync::Mutex<BTreeMap<u64, (Connection, Connection, wire::Hello)>>,
}

impl EngineControl {
    /// Binds the request socket at `socket` for the engines of `deployment`,
    /// which run on host `host` through the daemon whose
    /// process socket is `processes`.
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be bound.
    pub fn new(
        socket: &Path,
        processes: String,
        host: String,
        deployment: String,
    ) -> anyhow::Result<Self> {
        // A socket a stopped daemon left behind is replaced.
        let _ = std::fs::remove_file(socket);
        let address = if processes.starts_with("unix:") {
            format!("unix:{}", socket.display())
        } else {
            let endpoint: std::net::SocketAddr = processes.parse()?;
            std::net::SocketAddr::new(endpoint.ip(), 0).to_string()
        };
        let requests = Listener::bind(&address)
            .with_context(|| format!("binding the request socket {}", socket.display()))?;
        let request_address = requests.address().to_owned();
        Ok(Self {
            requests,
            request_address,
            socket: socket.to_path_buf(),
            processes,
            host,
            deployment,
            next_engine: AtomicU64::new(1),
            incoming: std::sync::Mutex::new(BTreeMap::new()),
        })
    }

    /// Starts an engine process that serves `spec` and returns its client
    /// once it reports that it serves, or, for a spec that stands by for the
    /// running engine (`EngineSpec::standby`), once it reports that it
    /// stands by; `stats` receives its schedulers' statistics. The process
    /// runs the executable of the spec's build (`workers.executable`).
    /// Blocks for the engine's whole start, its workers' loading included.
    ///
    /// # Errors
    ///
    /// Fails when the daemon cannot start the process, the process exits or
    /// does not connect back, its build speaks another major version, or it
    /// reports that its start failed.
    pub fn start(
        &self,
        mut spec: wire::EngineSpec,
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
        let engine = self.next_engine.fetch_add(1, Ordering::Relaxed);
        // A spec of no world starts a world of its own, which takes the
        // identity of its first engine; an engine that replaces a lost one
        // goes on with its world.
        if spec.world == 0 {
            spec.world = engine;
        }
        let world = spec.world;
        let executable = spec
            .workers
            .as_ref()
            .map(|workers| workers.executable.clone())
            .context("the engine spec states no workers")?;
        let standby = spec.predecessor.is_some();
        let mut control = self.spawn(engine, &executable, &spec.host)?;
        let (mut reader, mut writer, hello) = self.accept(engine, Some(&mut control))?;
        let refusal = if hello.major != proto::PROTOCOL_MAJOR {
            format!(
                "the engine speaks major version {} and this daemon {}",
                hello.major,
                proto::PROTOCOL_MAJOR
            )
        } else if hello.engine != engine {
            format!("engine {} connected for engine {engine}", hello.engine)
        } else {
            String::new()
        };
        writer.send(&wire::Welcome {
            major: proto::PROTOCOL_MAJOR,
            minor: proto::PROTOCOL_MINOR,
            build: proto::BUILD.to_owned(),
            refusal: refusal.clone(),
            spec: refusal.is_empty().then_some(spec),
        })?;
        anyhow::ensure!(refusal.is_empty(), "refused the engine process: {refusal}");
        tracing::info!(engine, build = %hello.build, "the engine process connected; awaiting its start");

        // The engine reports itself starting, then ready or failed. Its
        // process's exit, which the control connection reports, ends the
        // wait too.
        loop {
            match reader.try_recv::<wire::FromEngine>()? {
                Some(wire::FromEngine {
                    kind: Some(wire::from_engine::Kind::State(state)),
                }) => match wire::EngineState::try_from(state.state) {
                    Ok(started @ (wire::EngineState::Ready | wire::EngineState::Standby))
                        if (started == wire::EngineState::Standby) == standby =>
                    {
                        let info = state.info.context("the engine reported no info")?;
                        let state = if standby {
                            tracing::info!(engine, "the engine is prepared beside its predecessor");
                            EngineState::Standby
                        } else {
                            tracing::info!(
                                engine,
                                schedulers = info.schedulers,
                                "the engine serves"
                            );
                            EngineState::Ready
                        };
                        return EngineClient::new(
                            reader,
                            writer,
                            info.into(),
                            state,
                            stats,
                            // The process lives as long as its control
                            // connection.
                            Some(control),
                        )
                        .map(|client| client.numbered(engine).in_world(world));
                    }
                    Ok(wire::EngineState::Failed) => {
                        // The failed engine closes its workers and exits;
                        // their storage is free once it has.
                        await_exit(&mut control, FAILED_EXIT);
                        anyhow::bail!("the engine failed to start: {}", state.error);
                    }
                    _ => {}
                },
                Some(_) => {}
                None => {
                    exited(&mut control)?;
                    reader.wait(POLL_INTERVAL)?;
                }
            }
        }
    }

    /// The request socket's address, which an engine attaches to
    /// (`EngineClient::attach`).
    pub fn request_address(&self) -> &str {
        &self.request_address
    }

    /// The daemon process interface used to launch planning ranks on their nodes.
    pub(crate) fn process_address(&self) -> &str {
        &self.processes
    }

    /// The identity the next engine this control starts takes.
    pub fn next_engine(&self) -> u64 {
        self.next_engine.load(Ordering::Relaxed)
    }

    /// Takes over the engine `engine` a daemon this one replaced started,
    /// which serves `served` and lives under `control`: the engine connects
    /// to this daemon's request socket, as the replaced daemon commanded it
    /// (`EngineClient::attach`), is welcomed, and reports itself ready.
    /// Later engines this control starts take the identities from
    /// `next_engine` on. `stats` receives its schedulers' statistics.
    /// With no control connection, this is a borrowed front: another daemon
    /// retains process ownership and performs recovery.
    ///
    /// # Errors
    ///
    /// Fails when the engine does not connect in time, speaks another major
    /// version, is another engine, or reports anything but ready.
    pub fn adopt(
        &self,
        engine: u64,
        next_engine: u64,
        served: wire::EngineSpec,
        mut control: Option<Connection>,
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
        self.next_engine.fetch_max(next_engine, Ordering::Relaxed);
        let world = served.world;
        let (mut reader, mut writer, hello) = self.accept(engine, control.as_mut())?;
        let refusal = if hello.major != proto::PROTOCOL_MAJOR {
            format!(
                "the engine speaks major version {} and this daemon {}",
                hello.major,
                proto::PROTOCOL_MAJOR
            )
        } else if hello.engine != engine {
            format!(
                "engine {} attached in place of engine {engine}",
                hello.engine
            )
        } else {
            String::new()
        };
        writer.send(&wire::Welcome {
            major: proto::PROTOCOL_MAJOR,
            minor: proto::PROTOCOL_MINOR,
            build: proto::BUILD.to_owned(),
            refusal: refusal.clone(),
            spec: refusal.is_empty().then_some(served),
        })?;
        anyhow::ensure!(refusal.is_empty(), "refused the engine: {refusal}");
        let deadline = Instant::now() + CONNECT_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            anyhow::ensure!(!remaining.is_zero(), "engine {engine} reported no state");
            let Some(message) = reader.recv::<wire::FromEngine>(remaining)? else {
                continue;
            };
            if let Some(wire::from_engine::Kind::State(state)) = message.kind {
                anyhow::ensure!(
                    state.state == wire::EngineState::Ready as i32,
                    "engine {engine} attached while it does not serve: {}",
                    state.error
                );
                let info = state.info.context("the engine reported no info")?;
                tracing::info!(engine, build = %hello.build, "the engine attached");
                return EngineClient::new(
                    reader,
                    writer,
                    info.into(),
                    EngineState::Ready,
                    stats,
                    control,
                )
                .map(|client| client.numbered(engine).in_world(world));
            }
        }
    }

    /// Connects to the daemon's process socket and has it start engine
    /// `engine` as `executable` on this node; returns the control connection
    /// the process lives under.
    fn connect(&self) -> anyhow::Result<(Connection, Vec<process::Node>)> {
        let mut control = Connection::connect(&self.processes)
            .with_context(|| format!("connecting to the daemon at {}", self.processes))?;
        control.send(&process::Hello {
            major: proto::PROTOCOL_MAJOR,
            minor: proto::PROTOCOL_MINOR,
            build: proto::BUILD.to_owned(),
            peer: Some(hello::Peer::Engine(process::Engine {
                deployment: self.deployment.clone(),
            })),
        })?;
        let welcome: process::Welcome = control
            .recv(DAEMON_TIMEOUT)?
            .context("the daemon did not answer the control connection")?;
        anyhow::ensure!(
            welcome.refusal.is_empty(),
            "the daemon refused the control connection: {}",
            welcome.refusal
        );

        Ok((control, welcome.nodes))
    }

    /// Queries each daemon for its CUDA-visible devices with no compute or
    /// graphics owner. These ordinals can be assigned to a successor without
    /// displacing another serving world.
    pub fn available_devices(&self) -> anyhow::Result<BTreeMap<String, Vec<u32>>> {
        let (mut connection, nodes) = self.connect()?;
        let mut devices = BTreeMap::new();
        for (index, node) in nodes.into_iter().enumerate() {
            connection.send(&process::Upstream {
                kind: Some(upstream::Kind::Request(process::Request {
                    id: index as u64 + 1,
                    kind: Some(request::Kind::Devices(process::DeviceQuery {
                        host: node.host.clone(),
                    })),
                })),
            })?;
            let answer: process::Downstream = connection
                .recv(DAEMON_TIMEOUT)?
                .context("the daemon did not report its available devices")?;
            let Some(downstream::Kind::Reply(answer)) = answer.kind else {
                anyhow::bail!("the daemon did not answer its device query");
            };
            anyhow::ensure!(
                answer.error.is_empty(),
                "device inventory for {}: {}",
                node.host,
                answer.error
            );
            let Some(reply::Kind::Devices(inventory)) = answer.kind else {
                anyhow::bail!("the daemon returned no device inventory");
            };
            devices.insert(node.host, inventory.available);
        }
        Ok(devices)
    }

    /// Reads host admission from the deployment's existing Storage Server
    /// claims, including target nodes with no serving worker yet.
    pub(crate) fn host_storage(
        &self,
        hosts: &[String],
    ) -> anyhow::Result<BTreeMap<String, storage::protocol::HostStorage>> {
        let (mut connection, _) = self.connect()?;
        let mut storage = BTreeMap::new();
        for (index, host) in hosts.iter().enumerate() {
            connection.send(&process::Upstream {
                kind: Some(upstream::Kind::Request(process::Request {
                    id: index as u64 + 1,
                    kind: Some(request::Kind::Ledger(process::LedgerQuery {
                        host: host.clone(),
                        deployment: self.deployment.clone(),
                    })),
                })),
            })?;
            let answer: process::Downstream = connection
                .recv(DAEMON_TIMEOUT)?
                .context("the daemon did not report its host storage")?;
            let Some(downstream::Kind::Reply(answer)) = answer.kind else {
                anyhow::bail!("the daemon did not answer its ledger query");
            };
            anyhow::ensure!(
                answer.error.is_empty(),
                "host storage for {host}: {}",
                answer.error
            );
            let Some(reply::Kind::Ledger(report)) = answer.kind else {
                anyhow::bail!("the daemon returned no storage ledger");
            };
            #[derive(serde::Deserialize)]
            struct Ledger {
                host_storage: storage::protocol::HostStorage,
            }
            storage.insert(
                host.clone(),
                serde_json::from_str::<Ledger>(&report.ledgers)?.host_storage,
            );
        }
        Ok(storage)
    }

    /// Publishes one commit decision at every deployment node. Publication
    /// is atomic within each Storage Server and idempotent across retries;
    /// the daemon retains its commit decision until every node has answered.
    pub fn publish_worlds(
        &self,
        worlds: &[u64],
        pending: &mut Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let (mut connection, nodes) = self.connect()?;
        let pending =
            pending.get_or_insert_with(|| nodes.into_iter().map(|node| node.host).collect());
        while let Some(host) = pending.first().cloned() {
            connection.send(&process::Upstream {
                kind: Some(upstream::Kind::Request(process::Request {
                    id: 1,
                    kind: Some(request::Kind::PublishWorlds(process::PublishWorlds {
                        host: host.clone(),
                        deployment: self.deployment.clone(),
                        worlds: worlds.to_vec(),
                    })),
                })),
            })?;
            let answer: process::Downstream = connection
                .recv(DAEMON_TIMEOUT)?
                .context("the daemon did not report world publication")?;
            let Some(downstream::Kind::Reply(answer)) = answer.kind else {
                anyhow::bail!("the daemon did not answer world publication");
            };
            anyhow::ensure!(
                answer.error.is_empty(),
                "publishing worlds at {host}: {}",
                answer.error
            );
            anyhow::ensure!(
                matches!(answer.kind, Some(reply::Kind::WorldsPublished(_))),
                "the daemon returned no publication result"
            );
            pending.remove(0);
        }
        Ok(())
    }

    fn spawn(&self, engine: u64, executable: &str, host: &str) -> anyhow::Result<Connection> {
        let (mut control, nodes) = self.connect()?;
        anyhow::ensure!(
            nodes.iter().any(|node| node.host == host),
            "engine host {host} has not joined the deployment"
        );
        anyhow::ensure!(
            host == self.host || !self.processes.starts_with("unix:"),
            "a remote engine needs the deployment's network control address"
        );

        let executable = executable.to_owned();
        control.send(&process::Upstream {
            kind: Some(upstream::Kind::Request(process::Request {
                id: 1,
                kind: Some(request::Kind::Spawn(process::Spawn {
                    host: host.to_owned(),
                    group: ENGINE_GROUP.to_owned(),
                    rank: 0,
                    generation: engine,
                    executable,
                    arguments: vec![
                        "engine".to_owned(),
                        "--requests".to_owned(),
                        self.request_address.clone(),
                        "--processes".to_owned(),
                        self.processes.clone(),
                        "--engine".to_owned(),
                        engine.to_string(),
                    ],
                    ..process::Spawn::default()
                })),
            })),
        })?;
        let deadline = Instant::now() + DAEMON_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            anyhow::ensure!(
                !remaining.is_zero(),
                "the daemon did not start the engine in time"
            );
            let Some(message) = control.recv::<process::Downstream>(remaining)? else {
                continue;
            };
            if let Some(downstream::Kind::Reply(answer)) = message.kind {
                anyhow::ensure!(
                    answer.error.is_empty(),
                    "starting the engine process: {}",
                    answer.error
                );
                if let Some(reply::Kind::Spawned(spawned)) = answer.kind {
                    tracing::info!(engine, pid = spawned.pid, "started the engine process");
                }
                return Ok(control);
            }
        }
    }

    /// Accepts engine `engine`'s connection to the request socket, failing
    /// when its process exits first or does not connect in time.
    fn accept(
        &self,
        engine: u64,
        mut control: Option<&mut Connection>,
    ) -> anyhow::Result<(Connection, Connection, wire::Hello)> {
        let mut incoming = self
            .incoming
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(connection) = incoming.remove(&engine) {
            return Ok(connection);
        }
        let deadline = Instant::now() + CONNECT_TIMEOUT;
        loop {
            match self.requests.try_accept() {
                Ok(Some(mut reader)) => {
                    let writer = reader.writer()?;
                    let hello: wire::Hello = reader
                        .recv(CONNECT_TIMEOUT)?
                        .context("the engine said no hello")?;
                    if hello.engine == engine {
                        return Ok((reader, writer, hello));
                    }
                    incoming.insert(hello.engine, (reader, writer, hello));
                }
                Ok(None) => {}
                Err(error) => return Err(error).context("accepting the engine's connection"),
            }
            if let Some(control) = control.as_deref_mut() {
                exited(control)?;
            }
            anyhow::ensure!(
                Instant::now() < deadline,
                "engine {engine} did not connect within {CONNECT_TIMEOUT:?}"
            );
            std::thread::sleep(POLL_INTERVAL);
        }
    }
}

impl Drop for EngineControl {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket);
    }
}

/// How long a failed engine takes to close its workers and exit, at most.
const FAILED_EXIT: Duration = Duration::from_secs(120);

/// Waits up to `timeout` for the daemon to report the engine process's exit
/// on its control connection; a wait that times out is logged.
fn await_exit(control: &mut Connection, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        match exited(control) {
            Ok(()) => std::thread::sleep(POLL_INTERVAL),
            Err(_) => return,
        }
    }
    tracing::warn!("the failed engine process did not exit within {timeout:?}");
}

/// Fails when the daemon reported the engine process's exit on its control
/// connection.
fn exited(control: &mut Connection) -> anyhow::Result<()> {
    while let Some(message) = control.try_recv::<process::Downstream>()? {
        if let Some(downstream::Kind::Event(process::Event {
            kind: Some(event::Kind::Exit(exit)),
        })) = message.kind
        {
            anyhow::bail!("the engine process exited: {}", exit.status);
        }
    }
    Ok(())
}

/// One host's Program Server and the targets that host's devices build for
/// (§5.1, §5.3, §6.3).
#[derive(Debug, Clone)]
pub struct ProgramServer {
    /// The host's identity in the deployment.
    pub host: String,
    /// Its Program Server's socket.
    pub socket: PathBuf,
    /// The targets its devices compile for, which `prepare` builds the
    /// epoch's modules for.
    pub targets: Vec<program_server::Target>,
}

/// Selects the programs `filled` names at every host's Program Server
/// (§5.1, §6.3):
///
/// The leader host selects each slot's backend at its latest installed
/// revision (`Selector::Latest`), which is the revision the epoch record
/// pins. Every other host is then asked for exactly those programs
/// (`Selector::Fixed`): the same revision names the same module on every
/// host, so a follower that lacks it fails the call rather than choosing
/// another.
///
/// The answer is `filled` with each program's revision and the parameters
/// the Program Server selected for it, which is what the epoch record keeps
/// and what `Precopy` sends to the main processes.
///
/// # Errors
///
/// Fails when a host has no Program Server in `servers`, when a Program
/// Server refuses or cannot build a module, and when a follower answers a
/// program other than the one the leader pinned.
pub fn prepare_epoch(
    servers: &[ProgramServer],
    leader: &str,
    filled: &crate::epoch::FilledPrograms,
    descriptors: &[program_server::Descriptors],
) -> anyhow::Result<crate::epoch::FilledPrograms> {
    let leader_server = servers
        .iter()
        .find(|server| server.host == leader)
        .with_context(|| format!("no Program Server for the leader host {leader}"))?;
    let pinned = prepare_at(leader_server, Selector::Latest, filled, descriptors)?;
    for server in servers {
        if server.host == leader {
            continue;
        }
        let selected = prepare_at(server, Selector::Fixed, filled, descriptors)?;
        anyhow::ensure!(
            selected == pinned,
            "the Program Server of {} selected other programs than the leader's",
            server.host
        );
    }
    Ok(pinned)
}

/// How one host's `prepare` names the program that fills an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Selector {
    /// The backend's latest installed revision.
    Latest,
    /// Exactly this program, revision and all.
    Fixed,
}

/// Runs `prepare` at one host's Program Server, selecting each slot the way
/// `selector` says, and answers the epoch's programs with the revisions and
/// parameters it selected.
fn prepare_at(
    server: &ProgramServer,
    selector: Selector,
    filled: &crate::epoch::FilledPrograms,
    descriptors: &[program_server::Descriptors],
) -> anyhow::Result<crate::epoch::FilledPrograms> {
    let policy_slot = program_server::ProgramSlot::all(POLICY_SLOT);
    let mut requests = Vec::with_capacity(filled.kernels.len() + 1);
    for (slot, program) in &filled.kernels {
        requests.push(program_server::ProgramRequest {
            slot: slot.clone(),
            selector: selected(selector, program),
        });
    }
    requests.push(program_server::ProgramRequest {
        slot: policy_slot.clone(),
        selector: selected(selector, &filled.policy),
    });
    let request = program_server::PrepareRequest {
        targets: server.targets.clone(),
        programs: requests,
        descriptors: descriptors.to_vec(),
    };
    let mut client = program_server::Client::connect(&server.socket).with_context(|| {
        format!(
            "connecting to the Program Server of {} at {}",
            server.host,
            server.socket.display()
        )
    })?;
    let mut progress = |step: program_server::Progress| {
        tracing::debug!(
            host = %server.host,
            module = %step.path,
            state = ?step.state,
            "preparing a program's module"
        );
    };
    let prepared = client
        .prepare(&request, &mut progress)
        .with_context(|| format!("preparing the epoch's programs at {}", server.host))?;
    let mut kernels = BTreeMap::new();
    let mut policy = None;
    for chosen in prepared.programs {
        if chosen.slot == policy_slot {
            policy = Some(chosen.program);
        } else {
            kernels.insert(chosen.slot, chosen.program);
        }
    }
    let policy = policy.context("the Program Server selected no scheduling program")?;
    anyhow::ensure!(
        kernels.len() == filled.kernels.len(),
        "the Program Server of {} selected programs for {} of the {} slots asked",
        server.host,
        kernels.len(),
        filled.kernels.len()
    );
    Ok(crate::epoch::FilledPrograms { kernels, policy })
}

/// The slot the scheduling program fills: the name the policy packages state
/// as their `op` (`crates/scheduling/policies/<backend>/package.toml`). The
/// scheduling interface is not an operator (§5.1): the epoch record keeps its
/// program beside the kernels' rather than among them, so `prepare` names it
/// by this slot and the daemon takes it back out.
const POLICY_SLOT: &str = "scheduling";

/// The selector a `prepare` request uses for `program`.
fn selected(selector: Selector, program: &program_server::Program) -> program_server::Selector {
    match selector {
        Selector::Latest => program_server::Selector::Latest {
            backend: program.backend.clone(),
            arguments: program.arguments.clone(),
        },
        Selector::Fixed => program_server::Selector::Fixed(program.clone()),
    }
}

/// One epoch's main process on a node, and the daemon's process connection
/// to it: the connection its `Report`s arrive on and the `Command`s go down
/// (§2.3, §7.3).
pub struct EpochProcess {
    /// The node the main process runs on.
    pub host: String,
    /// The daemon's connection to it.
    pub connection: Connection,
}

/// How long a main process takes to prepare a generation's programs: it
/// resolves and loads the modules, instantiates the policy, rebinds or
/// recaptures the graphs and shadows the replaced operators (§7.3 step 2).
const PRECOPY_TIMEOUT: Duration = Duration::from_secs(20 * 60);

/// How long a main process takes to switch, commit, revert or purge: each is
/// a boundary decision the main process makes at its next batch (§7.3 steps
/// 3-6).
const COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

/// The program layer of a change (§7.3): it moves the serving main processes
/// from the generation an epoch runs to the one `prepare` selected, keeps
/// the previous generation through the probation, and ends the probation.
///
/// The order is the design's: `Precopy` every main process and wait for each
/// `Precopied`, then `Switchover` and wait for `Switched`; from there the new
/// generation serves while the previous one is kept (`UpdateState::Probation`);
/// then `Commit` or `Revert { reason }`, and `Purge { generation }` once the
/// previous generation's last batch has run.
///
/// Every report the main processes send while the switch runs is read: a
/// fault, a stalled job, non-finite logits or a fatal policy revert fails
/// the update with the first such failure ([`Update::fail`]), and
/// acknowledgements are matched to the command that waits for them.
pub struct ProgramSwitch<'a> {
    processes: &'a mut [EpochProcess],
    update: crate::epoch::Update,
    /// The program generation each main process numbered for the precopy
    /// it answered, in the processes' order: the main process numbers its
    /// generations, the daemon sends them back in `Switchover` and `Purge`.
    /// Empty before a precopy.
    generations: Vec<u64>,
}

impl<'a> ProgramSwitch<'a> {
    /// A switch of `processes`, the serving epoch's main processes, for
    /// `update`.
    pub fn new(processes: &'a mut [EpochProcess], update: crate::epoch::Update) -> Self {
        Self {
            processes,
            update,
            generations: Vec::new(),
        }
    }

    /// The update as it stands.
    pub fn update(&self) -> &crate::epoch::Update {
        &self.update
    }

    /// Prepares the next program generation in every main process: the kernel
    /// programs and the scheduling program, resolved and loaded, the policy
    /// instantiated and trialled, the graphs rebound or recaptured and the
    /// replaced operators shadowed (§7.3 step 2). A main process that cannot
    /// prepare them answers `PrecopyFailed` and keeps serving the generation
    /// it runs, so a failure here changes nothing that serves.
    ///
    /// # Errors
    ///
    /// Fails with the first failure: a main process's `PrecopyFailed`, or a
    /// fault report it sent meanwhile.
    pub fn precopy(
        &mut self,
        programs: &BTreeMap<crate::epoch::KernelKey, crate::epoch::Program>,
        policy: &crate::epoch::Program,
    ) -> Result<(), crate::epoch::Failure> {
        let command = process::Command {
            kind: Some(command::Kind::Precopy(process::Precopy {
                kernels: programs
                    .iter()
                    .map(|(slot, program)| crate::epoch::kernel_to_wire(slot, program))
                    .collect(),
                policy: Some(crate::epoch::program_to_wire(policy)),
            })),
        };
        let Self {
            processes,
            update,
            generations,
        } = self;
        // Every process's own number, in the processes' order: a switchover
        // names the generation each one prepared.
        generations.clear();
        for process in processes.iter_mut() {
            send(process, command.clone())?;
            let generation = match await_report(process, update, PRECOPY_TIMEOUT, |report| {
                match report.kind.as_ref() {
                    Some(report::Kind::Precopied(precopied)) => {
                        Some(Ok(precopied.generation))
                    }
                    Some(report::Kind::PrecopyFailed(failed)) => {
                        Some(Err(crate::epoch::Failure {
                            kind: "precopy_failed".to_owned(),
                            reason: failed.reason.clone(),
                            device: None,
                        }))
                    }
                    _ => None,
                }
            }) {
                Ok(generation) => generation,
                Err(failure) => {
                    // What was prepared is not switchable as a set: the
                    // serving generation stays, and no generation moves.
                    generations.clear();
                    return Err(failure);
                }
            };
            generations.push(generation);
        }
        Ok(())
    }

    /// Switches every main process to the generation it prepared at its next
    /// batch boundary (§7.3 step 3): from there the update is on probation,
    /// the previous generation kept as its last resort (§7.3 step 4).
    ///
    /// # Errors
    ///
    /// Fails with the first failure: a fault report, a main process that
    /// does not answer, or a call before [`Self::precopy`].
    pub fn switchover(&mut self) -> Result<(), crate::epoch::Failure> {
        let Self {
            processes,
            update,
            generations,
        } = self;
        if generations.len() != processes.len() {
            return Err(crate::epoch::Failure {
                kind: "out_of_order".to_owned(),
                reason: "a switchover before every main process prepared a generation".to_owned(),
                device: None,
            });
        }
        for (process, &generation) in processes.iter_mut().zip(generations.iter()) {
            send(
                process,
                process::Command {
                    kind: Some(command::Kind::Switchover(process::Switchover { generation })),
                },
            )?;
            await_report(process, update, COMMAND_TIMEOUT, |report| {
                match report.kind.as_ref() {
                    Some(report::Kind::Switched(switched)) => {
                        Some(answered(report, switched.generation == generation))
                    }
                    _ => None,
                }
            })?;
        }
        Ok(())
    }

    /// Ends the probation in favour of generation `generation` (§7.3 step 5):
    /// the previous generation is no longer kept.
    ///
    /// # Errors
    ///
    /// Fails with the first failure, as [`Self::switchover`] does. A failure
    /// leaves the update on probation and [`Self::revert`] is what ends it.
    pub fn commit(&mut self) -> Result<(), crate::epoch::Failure> {
        let command = process::Command {
            kind: Some(command::Kind::Commit(process::Commit {})),
        };
        let Self {
            processes, update, ..
        } = self;
        for process in processes.iter_mut() {
            send(process, command.clone())?;
            await_report(process, update, COMMAND_TIMEOUT, |report| {
                matches!(report.kind.as_ref(), Some(report::Kind::Committed(_))).then_some(Ok(()))
            })?;
        }
        update.commit();
        Ok(())
    }

    /// Returns every main process to the previous generation with `reason`,
    /// which is why the update did not stand (§7.3 step 5). The update stays
    /// unavailable with its first failure ([`Update::unavailable`]), so
    /// applying it again takes `force`.
    ///
    /// # Errors
    ///
    /// Fails when a main process does not answer the revert. The update is
    /// recorded as reverted either way: what serves is the previous
    /// generation.
    pub fn revert(&mut self, reason: &str) -> Result<(), crate::epoch::Failure> {
        let command = process::Command {
            kind: Some(command::Kind::Revert(process::Revert {
                reason: reason.to_owned(),
            })),
        };
        let Self {
            processes, update, ..
        } = self;
        let mut outcome = Ok(());
        for process in processes.iter_mut() {
            let answered = send(process, command.clone()).and_then(|()| {
                await_report(process, update, COMMAND_TIMEOUT, |report| {
                    matches!(report.kind.as_ref(), Some(report::Kind::Reverted(_)))
                        .then_some(Ok(()))
                })
            });
            if let Err(failure) = answered {
                update.fail(failure.clone());
                outcome = Err(failure);
            }
        }
        update.revert();
        outcome
    }

    /// Drops everything older than the generation each main process executes
    /// once no batch uses it (§7.3 step 6): the previous generation's kernels'
    /// last reference unloads their libraries.
    ///
    /// # Errors
    ///
    /// Fails when a main process does not answer the purge, or when none was
    /// prepared.
    pub fn purge(&mut self) -> Result<(), crate::epoch::Failure> {
        let Self {
            processes,
            update,
            generations,
        } = self;
        if generations.len() != processes.len() {
            return Err(crate::epoch::Failure {
                kind: "out_of_order".to_owned(),
                reason: "a purge before every main process prepared a generation".to_owned(),
                device: None,
            });
        }
        for (process, &generation) in processes.iter_mut().zip(generations.iter()) {
            send(
                process,
                process::Command {
                    kind: Some(command::Kind::Purge(process::Purge { generation })),
                },
            )?;
            await_report(process, update, COMMAND_TIMEOUT, |report| {
                matches!(report.kind.as_ref(), Some(report::Kind::Purged(_))).then_some(Ok(()))
            })?;
        }
        Ok(())
    }
}

/// Sends `command` down the process connection to one main process.
fn send(
    process: &mut EpochProcess,
    command: process::Command,
) -> Result<(), crate::epoch::Failure> {
    process
        .connection
        .send(&process::Downstream {
            kind: Some(downstream::Kind::Event(process::Event {
                kind: Some(event::Kind::Command(command)),
            })),
        })
        .map_err(|error| unanswerable(&process.host, error))
}

/// Reads one main process's reports until `matches` takes one: the answer to
/// the command that waits, or the failure that ends the wait. A fault report
/// the main process sends meanwhile fails the update with its first failure
/// and ends the wait.
fn await_report<T>(
    process: &mut EpochProcess,
    update: &mut crate::epoch::Update,
    timeout: Duration,
    matches: impl Fn(&process::Report) -> Option<Result<T, crate::epoch::Failure>>,
) -> Result<T, crate::epoch::Failure> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let Some(upstream) = process
            .connection
            .recv::<process::Upstream>(remaining)
            .map_err(|error| unanswerable(&process.host, error))?
        else {
            return Err(crate::epoch::Failure {
                kind: "timeout".to_owned(),
                reason: format!(
                    "the main process of {} did not answer within {} s",
                    process.host,
                    timeout.as_secs()
                ),
                device: None,
            });
        };
        let Some(upstream::Kind::Report(report)) = upstream.kind else {
            // Requests from a main process are none of the switch's
            // business; the process service answers them.
            continue;
        };
        if let Some(outcome) = matches(&report) {
            if let Err(failure) = &outcome {
                update.fail(failure.clone());
            }
            return outcome;
        }
        if let Some(failure) = crate::epoch::Failure::of_report(&report) {
            update.fail(failure.clone());
            return Err(failure);
        }
    }
}

/// The outcome of a report that answers a command.
fn answered(report: &process::Report, matched: bool) -> Result<(), crate::epoch::Failure> {
    if matched {
        return Ok(());
    }
    Err(crate::epoch::Failure {
        kind: "out_of_order".to_owned(),
        reason: format!("the main process answered a generation the switch did not ask for: {report:?}"),
        device: None,
    })
}

/// The failure a process connection that cannot be written or read states.
fn unanswerable(host: &str, error: impl std::fmt::Display) -> crate::epoch::Failure {
    crate::epoch::Failure {
        kind: "process_exit".to_owned(),
        reason: format!("the main process of {host}: {error}"),
        device: None,
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixListener;
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::options::ServeConfig;

/// What a stand-in main process saw a daemon send it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Seen {
    Precopy,
    Switchover(u64),
    Commit,
    Revert(String),
    Purge(u64),
}

/// A stand-in main process on the other end of the daemon's process
/// connection: it records the commands it is sent and answers them as the
/// protocol says the main process answers, numbering a generation of its own
/// for the precopy it answers.
fn main_process(host: &str, generation: u64) -> (EpochProcess, Arc<Mutex<Vec<Seen>>>) {
    answer_as(host, move |command| match command.kind.as_ref() {
        Some(command::Kind::Precopy(_)) => Some((
            Seen::Precopy,
            Some(process::Report {
                kind: Some(report::Kind::Precopied(process::Precopied { generation })),
            }),
        )),
        Some(command::Kind::Switchover(switchover)) => Some((
            Seen::Switchover(switchover.generation),
            Some(process::Report {
                kind: Some(report::Kind::Switched(process::Switched {
                    generation: switchover.generation,
                })),
            }),
        )),
        Some(command::Kind::Commit(_)) => Some((
            Seen::Commit,
            Some(process::Report {
                kind: Some(report::Kind::Committed(process::Committed {})),
            }),
        )),
        Some(command::Kind::Revert(revert)) => Some((
            Seen::Revert(revert.reason.clone()),
            Some(process::Report {
                kind: Some(report::Kind::Reverted(process::Reverted {
                    reason: String::new(),
                })),
            }),
        )),
        Some(command::Kind::Purge(purge)) => Some((
            Seen::Purge(purge.generation),
            Some(process::Report {
                kind: Some(report::Kind::Purged(process::Purged {
                    generation: purge.generation,
                })),
            }),
        )),
        other => panic!("the daemon sent a command the switch does not send: {other:?}"),
    })
}

/// A stand-in main process whose precopy fails with `reason` and which
/// answers nothing else, as a main process that kept serving the generation
/// it runs.
fn refusing(host: &str, reason: &str) -> (EpochProcess, Arc<Mutex<Vec<Seen>>>) {
    let reason = reason.to_owned();
    answer_as(host, move |command| match command.kind.as_ref() {
        Some(command::Kind::Precopy(_)) => Some((
            Seen::Precopy,
            Some(process::Report {
                kind: Some(report::Kind::PrecopyFailed(process::PrecopyFailed {
                    reason: reason.clone(),
                })),
            }),
        )),
        other => panic!("a failed precopy is not followed by a command: {other:?}"),
    })
}

/// A stand-in main process that reports `report` when the precopy arrives.
fn reporting(
    host: &str,
    report: impl Fn() -> process::Report + Send + 'static,
) -> (EpochProcess, Arc<Mutex<Vec<Seen>>>) {
    answer_as(host, move |command| match command.kind.as_ref() {
        Some(command::Kind::Precopy(_)) => Some((Seen::Precopy, Some(report()))),
        other => panic!("a failing report is not followed by a command: {other:?}"),
    })
}

/// The daemon's end of a connection to a stand-in main process that records
/// what it is sent and answers with `answer`.
fn answer_as(
    host: &str,
    answer: impl Fn(&process::Command) -> Option<(Seen, Option<process::Report>)> + Send + 'static,
) -> (EpochProcess, Arc<Mutex<Vec<Seen>>>) {
    let (daemon, peer) = std::os::unix::net::UnixStream::pair().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&seen);
    std::thread::spawn(move || {
        let mut peer = Connection::from_unix(peer).unwrap();
        while let Ok(Some(message)) = peer.recv::<process::Downstream>(Duration::from_secs(60)) {
            let Some(downstream::Kind::Event(process::Event {
                kind: Some(event::Kind::Command(command)),
            })) = message.kind
            else {
                continue;
            };
            let Some((what, answer)) = answer(&command) else {
                continue;
            };
            recorded.lock().unwrap().push(what);
            if let Some(report) = answer {
                peer.send(&process::Upstream {
                    kind: Some(upstream::Kind::Report(report)),
                })
                .unwrap();
            }
        }
    });
    let process = EpochProcess {
        host: host.to_owned(),
        connection: Connection::from_unix(daemon).unwrap(),
    };
    (process, seen)
}

/// An update of the default deployment's filled programs, for the switch to
/// move.
fn update() -> crate::epoch::Update {
    crate::epoch::Update::new(1, crate::epoch::Deployment::of(ServeConfig::default()))
}

/// The program layer moves every main process through the generations each
/// one numbered for itself (§7.3): the precopy, then the switchover to the
/// generation that process answered with, then the commit, and the purge of
/// the generation each one executes.
#[test]
fn a_switch_moves_every_main_process_through_the_generations_they_numbered() {
    let (first, first_seen) = main_process("rank-0", 11);
    let (second, second_seen) = main_process("rank-1", 12);
    let mut processes = vec![first, second];
    let target = crate::epoch::Deployment::of(ServeConfig::default());
    let mut switch = ProgramSwitch::new(&mut processes, update());
    switch
        .precopy(&target.programs.kernels, &target.programs.policy)
        .expect("both main processes prepared the programs");
    switch.switchover().expect("both switched to what they prepared");
    switch.commit().expect("both committed");
    switch.purge().expect("both purged what they no longer run");
    assert_eq!(switch.update().state, crate::epoch::UpdateState::Committed);

    let seen = |seen: &Arc<Mutex<Vec<Seen>>>| seen.lock().unwrap().clone();
    // Each process's own number comes back to it: the daemon does not
    // invent one.
    assert_eq!(
        seen(&first_seen),
        vec![
            Seen::Precopy,
            Seen::Switchover(11),
            Seen::Commit,
            Seen::Purge(11)
        ]
    );
    assert_eq!(
        seen(&second_seen),
        vec![
            Seen::Precopy,
            Seen::Switchover(12),
            Seen::Commit,
            Seen::Purge(12)
        ]
    );
}

/// A main process that cannot prepare the programs fails the update and
/// keeps serving the generation it runs: the serving generation is never
/// switched, and the update is not unavailable, since nothing reverted
/// (§7.3 step 2).
#[test]
fn a_failed_precopy_ends_the_switch_before_any_generation_moves() {
    let (fine, fine_seen) = main_process("rank-0", 11);
    let (failing, _) = refusing("rank-1", "the package has no module for this GPU");
    let mut processes = vec![fine, failing];
    let target = crate::epoch::Deployment::of(ServeConfig::default());
    let mut switch = ProgramSwitch::new(&mut processes, update());
    let failure = switch
        .precopy(&target.programs.kernels, &target.programs.policy)
        .expect_err("a refused precopy fails");
    assert_eq!(failure.kind, "precopy_failed");
    assert!(
        failure.reason.contains("no module for this GPU"),
        "{}",
        failure.reason
    );
    // Nothing moved: the first process prepared and was never switched.
    assert_eq!(*fine_seen.lock().unwrap(), vec![Seen::Precopy]);
    assert_eq!(switch.update().state, crate::epoch::UpdateState::Probation);
    assert!(switch.update().unavailable().is_none());
    // A switchover without a prepared generation is refused, not sent.
    assert!(switch.switchover().is_err());
}

/// A main process's fault while the programs are prepared fails the update
/// with that first failure, and a reverted update stays unavailable until
/// `force` is asked for (§7.3 step 4).
#[test]
fn a_fault_during_the_precopy_is_the_updates_first_failure() {
    let (faulty, _) = reporting("rank-0", || process::Report {
        kind: Some(report::Kind::Fault(process::Fault {
            kind: "sticky".to_owned(),
            reason: "an earlier launch failed".to_owned(),
            device: 1,
        })),
    });
    let mut processes = vec![faulty];
    let target = crate::epoch::Deployment::of(ServeConfig::default());
    let mut switch = ProgramSwitch::new(&mut processes, update());
    let failure = switch
        .precopy(&target.programs.kernels, &target.programs.policy)
        .expect_err("a fault fails the update");
    assert_eq!(failure.kind, "sticky");
    switch
        .revert(&failure.reason)
        .expect("the reverted main process answers");
    assert_eq!(switch.update().state, crate::epoch::UpdateState::Reverted);
    let unavailable = switch
        .update()
        .unavailable()
        .expect("a reverted update is unavailable");
    assert_eq!(unavailable.kind, "sticky");
}

/// A policy that fails after the switchover reports `PolicyReverted` without
/// being asked, and that report is the update's first failure (§4.4).
#[test]
fn a_policy_revert_while_the_programs_are_prepared_fails_the_update() {
    let (reverted, _) = reporting("rank-0", || process::Report {
        kind: Some(report::Kind::PolicyReverted(process::PolicyReverted {
            scheduler: Some(0),
            backend: "priority".to_owned(),
            revision: 3,
            reason: "the trial refused a decision".to_owned(),
            fatal: false,
        })),
    });
    let mut processes = vec![reverted];
    let target = crate::epoch::Deployment::of(ServeConfig::default());
    let mut switch = ProgramSwitch::new(&mut processes, update());
    let failure = switch
        .precopy(&target.programs.kernels, &target.programs.policy)
        .expect_err("a policy revert fails the update");
    assert_eq!(failure.kind, "policy_reverted");
    assert!(
        failure.reason.contains("priority"),
        "{}",
        failure.reason
    );
}


    #[test]
    fn publication_resumes_after_a_node_refuses_the_commit() {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("processes.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        // This process-protocol peer represents two remote storage nodes.
        // One node cannot publish until the operator restores its storage.
        let peer = std::thread::spawn(move || {
            let mut published = BTreeMap::new();
            for restored in [false, true] {
                let mut connection = Connection::from_unix(listener.accept().unwrap().0).unwrap();
                let _: process::Hello = connection.recv(Duration::from_secs(5)).unwrap().unwrap();
                connection
                    .send(&process::Welcome {
                        nodes: ["rank-0", "rank-1"]
                            .into_iter()
                            .map(|host| process::Node {
                                host: host.into(),
                                ..Default::default()
                            })
                            .collect(),
                        ..Default::default()
                    })
                    .unwrap();
                while let Ok(Some(message)) =
                    connection.recv::<process::Upstream>(Duration::from_secs(5))
                {
                    let Some(upstream::Kind::Request(command)) = message.kind else {
                        panic!("expected a request")
                    };
                    let Some(request::Kind::PublishWorlds(worlds)) = command.kind else {
                        panic!("expected world publication")
                    };
                    let unavailable = worlds.host == "rank-1" && !restored;
                    if !unavailable {
                        published.insert(worlds.host, worlds.worlds);
                    }
                    connection
                        .send(&process::Downstream {
                            kind: Some(downstream::Kind::Reply(process::Reply {
                                id: command.id,
                                error: if unavailable {
                                    "storage unavailable".into()
                                } else {
                                    String::new()
                                },
                                kind: (!unavailable).then_some(reply::Kind::WorldsPublished(
                                    process::WorldsPublished {},
                                )),
                            })),
                        })
                        .unwrap();
                }
            }
            published
        });
        let control = EngineControl::new(
            &directory.path().join("requests.sock"),
            format!("unix:{}", socket.display()),
            "rank-0".into(),
            "model".into(),
        )
        .unwrap();
        let mut pending = None;
        assert!(control.publish_worlds(&[3, 4], &mut pending).is_err());
        assert_eq!(pending, Some(vec!["rank-1".to_owned()]));
        // Handoff serializes the pending nodes with the irrevocable decision.
        let mut handed = serde_json::from_slice(&serde_json::to_vec(&pending).unwrap()).unwrap();
        control.publish_worlds(&[3, 4], &mut handed).unwrap();
        assert_eq!(handed, Some(Vec::new()));
        assert_eq!(
            peer.join().unwrap(),
            BTreeMap::from([
                ("rank-0".to_owned(), vec![3, 4]),
                ("rank-1".to_owned(), vec![3, 4]),
            ])
        );
    }
}
