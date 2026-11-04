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
use crate::engine::launch::{self, Launch};

/// The worker group a daemon supervises an engine process under.
pub(crate) const ENGINE_GROUP: &str = "engine";

/// How long an engine process takes to connect back to the request socket
/// once the daemon started it.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// How long the daemon takes to answer the control connection.
const DAEMON_TIMEOUT: Duration = Duration::from_secs(90);

/// How often a wait rechecks the engine process while it starts.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Starts the engine processes of one deployment.
/// Whether `executable` is this process's own binary, by canonical path: a
/// bare name is looked up by the operating system, not by us, and counts as
/// another executable.
fn own_executable(executable: &str) -> bool {
    let path = std::path::Path::new(executable);
    if !path.is_absolute() {
        return false;
    }
    let Ok(own) = std::env::current_exe() else {
        return false;
    };
    let own = std::fs::canonicalize(&own).unwrap_or(own);
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()) == own
}

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
    /// This node's Program Server socket, which every engine process is told
    /// where to resolve an epoch's policy modules from (§5.3, §7.3); `None`
    /// leaves an engine with no policy store, and it refuses every policy
    /// precopy with a reason rather than serving without one.
    program_server: Option<PathBuf>,
    /// Where this daemon writes the files its main processes read: the
    /// epoch's build config and each process's launch (§2.3).
    runtime: PathBuf,
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
        program_server: Option<PathBuf>,
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
            program_server,
            runtime: crate::options::runtime_directory(),
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
        spec: wire::EngineSpec,
        build: &crate::epoch::BuildConfig,
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
        let engine = self.next_engine.fetch_add(1, Ordering::Relaxed);
        self.start_at(spec, engine, build, stats)
    }

    /// [`EngineControl::start`] under a number the caller allocated.
    ///
    /// A standby's number comes from the epoch layer's one allocator
    /// ([`crate::epoch::Epochs::next_generation`]), which also numbers the
    /// launches an epoch switch starts; two counters for one identity would
    /// drift the day a standby exists, so the standby path does not use
    /// `next_engine`.
    ///
    /// # Errors
    ///
    /// As [`EngineControl::start`].
    pub fn start_at(
        &self,
        mut spec: wire::EngineSpec,
        engine: u64,
        build: &crate::epoch::BuildConfig,
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
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
        // The epoch's build config, written once per world: every process of
        // that world reads the same bytes, and the launch file names its path
        // (`specs/worker.md` §2.3).
        self.write_build_config(world, build)?;
        let launch = self.launch(&spec, engine)?;
        let path = launch::write(&self.runtime, engine, &launch)?;
        let mut control = self.spawn(engine, &executable, &spec.host, Self::arguments(&path))?;
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
    /// The host this control's engines run on.
    pub fn host(&self) -> &str {
        &self.host
    }

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

    /// The arguments of a main process: its program name and the launch file
    /// the daemon wrote for it, and nothing else. Everything the process needs
    /// is in that file, so a flag beside it would be a second account of the
    /// same thing (`specs/worker.md` §2.3).
    fn arguments(launch: &Path) -> Vec<String> {
        vec![
            "engine".to_owned(),
            "--launch".to_owned(),
            launch.display().to_string(),
        ]
    }

    /// The launch file's contents for the process `engine`, from the spec the
    /// deployment resolved for it.
    ///
    /// A value the file requires and the daemon cannot state is an error here
    /// rather than a default: the engine refuses a launch file that lacks one,
    /// and a default would be a fact the daemon invented about the process.
    ///
    /// # Errors
    ///
    /// Fails for a spec of no model, no placement or a zero queue depth, a
    /// child on a host that does not share this daemon's runtime directory, a
    /// node with no Program Server, a spec of no world, and an epoch whose
    /// build config has not been written.
    fn launch(&self, spec: &wire::EngineSpec, engine: u64) -> anyhow::Result<Launch> {
        anyhow::ensure!(!spec.model.is_empty(), "the engine spec names no model");
        // Form (A): the child opens the launch file this daemon wrote, so it
        // has to be this daemon's own host. A remote host needs the launch
        // contents carried in `process::Spawn`, which the protocol does not do
        // yet, and a path that exists only here would fail in the child.
        anyhow::ensure!(
            spec.host == self.host,
            "engine host {} does not share this daemon's runtime directory {}: a main process on \
             another host needs the launch contents carried in `process::Spawn`",
            spec.host,
            self.runtime.display()
        );
        let placement = spec
            .placement
            .as_ref()
            .context("the engine spec states no placement")?;
        // The devices are the deployment's own: each host's ranks are on the
        // ordinals the engine's shorthand gives them (`0..` in rank order), or
        // on the order `placement_config.devices` names, which
        // `EngineConfig::from_spec` has already folded into the ranks. Reading
        // them from the ranks is the one account: a config that states no
        // order still gives this node's ranks their devices, and a rank on
        // `cpu` is refused here rather than in the child.
        let devices = local_devices(&self.host, spec)?;
        anyhow::ensure!(
            placement.queue_depth > 0,
            "the placement's queue_depth is zero for this rank"
        );
        let program_server = self.program_server.as_ref().context(
            "this node has no Program Server socket, so a main process could resolve no policy \
             module: it is not started without one",
        )?;
        anyhow::ensure!(spec.world != 0, "the engine spec names no world");
        let build_config = launch::build_config_path(&self.runtime, spec.world);
        anyhow::ensure!(
            build_config.is_file(),
            "the epoch's build config {} has not been written: write it with \
             `EngineControl::write_build_config` before starting a main process",
            build_config.display()
        );
        // One main process per node today, so it has no peer to accept: the
        // world is this rank alone. A deployment that starts more states their
        // addresses here, and the world grows with them (exec's rule: the
        // world is its peers and itself).
        let peers: Vec<String> = Vec::new();
        let world_size = u32::try_from(peers.len() + 1).unwrap_or(u32::MAX);
        Ok(Launch {
            build_config,
            model: PathBuf::from(&spec.model),
            requests: self.request_address.clone(),
            processes: self.processes.clone(),
            engine,
            program_server: program_server.display().to_string(),
            devices,
            group: ENGINE_GROUP.to_owned(),
            rank: 0,
            deployment: self.deployment.clone(),
            world: spec.world,
            // The check above makes this node the engine's host, and the
            // deployment keeps one scheduler node, which is this one.
            scheduler_node: true,
            peers,
            // Both are the epoch's, and the epoch states them in its build
            // config, which the child reads.
            watchdog_timeout_s: None,
            rollback_s: None,
            queue_depth: placement.queue_depth,
            world_size,
            // A process that starts beside its predecessor is the serving
            // process's standby (§2.5): the spec states the predecessor and
            // the child has to know the role before it claims storage.
            standby: spec.predecessor.is_some(),
        })
    }

    /// Writes the config `config` builds the epoch whose world is `world`, and
    /// answers the path every process of that epoch reads.
    ///
    /// Whoever owns the epoch's record calls this before starting any of its
    /// processes: the launch file names the path, and two engines of one epoch
    /// read the same bytes.
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be written.
    pub(crate) fn write_build_config(
        &self,
        world: u64,
        config: &crate::epoch::BuildConfig,
    ) -> anyhow::Result<PathBuf> {
        Ok(launch::write_build_config(&self.runtime, world, config)?)
    }

    /// What the epoch layer starts one epoch's main process with: the
    /// executable the target names, at `generation`, and the arguments every
    /// engine process of this daemon gets (`specs/worker.md` §7.4).
    ///
    /// The build is named only when `executable` is this daemon's own binary.
    /// Another executable's children vouch for themselves: the first one's
    /// `Hello` fixes the identity every other child of the epoch must report.
    pub(crate) fn epoch_start(
        &self,
        spec: &wire::EngineSpec,
        engine: u64,
    ) -> anyhow::Result<crate::updates::switch::Start> {
        let launch = self.launch(spec, engine)?;
        let executable = spec
            .workers
            .as_ref()
            .map(|workers| workers.executable.clone())
            .context("the engine spec states no workers")?;
        Ok(crate::updates::switch::Start {
            build: own_executable(&executable).then(|| proto::BUILD.to_owned()),
            executable,
            launch,
            environment: std::collections::HashMap::new(),
        })
    }

    fn spawn(
        &self,
        engine: u64,
        executable: &str,
        host: &str,
        arguments: Vec<String>,
    ) -> anyhow::Result<Connection> {
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
                    arguments,
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
    // A program is selected by its backend's name, so a deployment that left
    // one at `auto` has not said which program it means: refused here, by the
    // option that would state it, rather than as a package the server is not
    // missing (§5.1).
    filled.stated().map_err(anyhow::Error::msg)?;
    let leader_server = servers
        .iter()
        .find(|server| server.host == leader)
        .with_context(|| format!("no Program Server for the leader host {leader}"))?;
    tracing::info!(
        leader = %leader,
        servers = servers.len(),
        kernels = filled.kernels.len(),
        policy = %filled.policy.backend,
        "resolving the epoch's programs at its Program Servers"
    );
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
    tracing::info!(
        kernels = pinned.kernels.len(),
        policy = %pinned.policy.backend,
        "the epoch's programs are resolved"
    );
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
    // Before the connect, not after it: a daemon that blocks opening the
    // socket names the host it was reaching from this line alone, and the
    // server it never reached logs nothing at all.
    tracing::info!(
        host = %server.host,
        socket = %server.socket.display(),
        "connecting to a Program Server"
    );
    let mut client = program_server::Client::connect(&server.socket).with_context(|| {
        format!(
            "connecting to the Program Server of {} at {}",
            server.host,
            server.socket.display()
        )
    })?;
    tracing::info!(
        host = %server.host,
        socket = %server.socket.display(),
        targets = ?server.targets,
        programs = request.programs.len(),
        "asking a Program Server to prepare the epoch's programs"
    );
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
    tracing::info!(
        host = %server.host,
        prepared = prepared.programs.len(),
        "a Program Server answered"
    );
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
    /// The engine number its process connection is registered under, which a
    /// command names and a report carries.
    pub engine: u64,
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
    /// The daemon's command channel to the main processes (§2.3).
    commands: &'a crate::supervision::ProcessCommands,
    processes: Vec<EpochProcess>,
    update: crate::epoch::Update,
    /// The program generation each main process numbered for the precopy it
    /// answered, in the processes' order: the main process numbers its
    /// generations, the daemon sends them back in `Switchover` and `Purge`.
    /// Empty before a precopy.
    generations: Vec<u64>,
}

impl<'a> ProgramSwitch<'a> {
    /// A switch of `processes`, the serving epoch's main processes, for
    /// `update`, commanded through the daemon's channel to them.
    pub fn new(
        commands: &'a crate::supervision::ProcessCommands,
        processes: Vec<EpochProcess>,
        update: crate::epoch::Update,
    ) -> Self {
        Self {
            commands,
            processes,
            update,
            generations: Vec::new(),
        }
    }

    /// A switch that goes on from a precopy whose generations the daemon
    /// recorded: the commit, the revert or the purge of an update that is on
    /// probation (§7.3 steps 4-6).
    pub fn resuming(
        commands: &'a crate::supervision::ProcessCommands,
        processes: Vec<EpochProcess>,
        update: crate::epoch::Update,
        generations: Vec<u64>,
    ) -> Self {
        Self {
            commands,
            processes,
            update,
            generations,
        }
    }

    /// The update as it stands.
    pub fn update(&self) -> &crate::epoch::Update {
        &self.update
    }

    /// The generation each main process prepared, in the processes' order.
    pub fn generations(&self) -> &[u64] {
        &self.generations
    }

    /// The main processes the switch commands, in the generations' order.
    pub fn processes(&self) -> &[EpochProcess] {
        &self.processes
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
            commands,
            processes,
            update,
            generations,
        } = self;
        // Every process's own number, in the processes' order: a switchover
        // names the generation each one prepared.
        generations.clear();
        for process in processes.iter() {
            send(commands, process, command.clone(), "Precopy")?;
            let prepared = await_report(
                commands,
                process,
                update,
                PRECOPY_TIMEOUT,
                "Precopy",
                |report| match report.kind.as_ref() {
                    Some(report::Kind::Precopied(precopied)) => Some(Ok(precopied.generation)),
                    Some(report::Kind::PrecopyFailed(failed)) => Some(Err(crate::epoch::Failure {
                        kind: "precopy_failed".to_owned(),
                        reason: failed.reason.clone(),
                        device: None,
                    })),
                    _ => None,
                },
            );
            match prepared {
                Ok(generation) => generations.push(generation),
                Err(failure) => {
                    // What was prepared is not switchable as a set: the
                    // serving generation stays, and no generation moves.
                    generations.clear();
                    return Err(failure);
                }
            }
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
        self.prepared()?;
        for (process, generation) in self.processes.iter().zip(self.generations.clone()) {
            send(
                self.commands,
                &process,
                process::Command {
                    kind: Some(command::Kind::Switchover(process::Switchover {
                        generation,
                    })),
                },
                "Switchover",
            )?;
            await_report(
                self.commands,
                &process,
                &mut self.update,
                COMMAND_TIMEOUT,
                "Switchover",
                |report| match report.kind.as_ref() {
                    Some(report::Kind::Switched(switched)) => {
                        Some(answered(report, switched.generation == generation))
                    }
                    _ => None,
                },
            )?;
        }
        Ok(())
    }

    /// Ends the probation in favour of the generation each main process
    /// executes (§7.3 step 5): the previous generation is no longer kept.
    ///
    /// # Errors
    ///
    /// Fails with the first failure, as [`Self::switchover`] does. A failure
    /// leaves the update on probation and [`Self::revert`] is what ends it.
    pub fn commit(&mut self) -> Result<(), crate::epoch::Failure> {
        let command = process::Command {
            kind: Some(command::Kind::Commit(process::Commit {})),
        };
        let mut acknowledged = 0;
        for process in self.processes.iter() {
            send(self.commands, process, command.clone(), "Commit")?;
            await_report(
                self.commands,
                process,
                &mut self.update,
                COMMAND_TIMEOUT,
                "Commit",
                |report| {
                    matches!(report.kind.as_ref(), Some(report::Kind::Committed(_)))
                        .then_some(Ok(()))
                },
            )?;
            acknowledged += 1;
        }
        debug_assert_eq!(acknowledged, self.processes.len());
        self.update.commit();
        Ok(())
    }

    /// Returns every main process to the previous generation with `reason`,
    /// which is why the update did not stand (§7.3 step 5). The update stays
    /// unavailable with its first failure, so applying it again takes `force`.
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
        let mut outcome = Ok(());
        for process in self.processes.iter() {
            let answered = send(self.commands, process, command.clone(), "Revert").and_then(|()| {
                await_report(
                    self.commands,
                    process,
                    &mut self.update,
                    COMMAND_TIMEOUT,
                    "Revert",
                    |report| {
                        matches!(report.kind.as_ref(), Some(report::Kind::Reverted(_)))
                            .then_some(Ok(()))
                    },
                )
            });
            if let Err(failure) = answered {
                self.update.fail(failure.clone());
                outcome = Err(failure);
            }
        }
        self.update.revert();
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
        self.prepared()?;
        for (process, generation) in self.processes.iter().zip(self.generations.clone()) {
            send(
                self.commands,
                &process,
                process::Command {
                    kind: Some(command::Kind::Purge(process::Purge { generation })),
                },
                "Purge",
            )?;
            await_report(
                self.commands,
                &process,
                &mut self.update,
                COMMAND_TIMEOUT,
                "Purge",
                |report| {
                    matches!(report.kind.as_ref(), Some(report::Kind::Purged(_))).then_some(Ok(()))
                },
            )?;
        }
        Ok(())
    }

    /// Fails when the generations are not the ones a precopy answered, which
    /// a switchover and a purge need.
    fn prepared(&self) -> Result<(), crate::epoch::Failure> {
        if self.generations.len() == self.processes.len() {
            return Ok(());
        }
        Err(crate::epoch::Failure {
            kind: "out_of_order".to_owned(),
            reason: "no main process prepared a generation for this step".to_owned(),
            device: None,
        })
    }
}

/// Sends `command` down the daemon's channel to one main process.
fn send(
    commands: &crate::supervision::ProcessCommands,
    process: &EpochProcess,
    command: process::Command,
    what: &str,
) -> Result<(), crate::epoch::Failure> {
    // The epoch's installation is the one sequence whose silence cannot be
    // told from a stall: a log line here says which command a daemon was
    // sending when nothing answered any more (§7.3 steps 2-6).
    tracing::info!(
        host = %process.host,
        engine = process.engine,
        command = what,
        "sending a command to a main process"
    );
    let sent = commands
        .send(process.engine, command)
        .map_err(|error| unanswerable(&process.host, error));
    if sent.is_ok() {
        tracing::info!(
            host = %process.host,
            engine = process.engine,
            command = what,
            "the command is sent"
        );
    }
    sent
}

/// Reads the reports the main processes send until `matches` takes one from
/// this process: the answer to the command that waits, or the failure that
/// ends the wait. A failure report from any main process fails the update
/// with the first such failure (§7.3 step 4); an acknowledgement for another
/// process's command is not this wait's.
fn await_report<T>(
    commands: &crate::supervision::ProcessCommands,
    process: &EpochProcess,
    update: &mut crate::epoch::Update,
    timeout: Duration,
    what: &str,
    matches: impl Fn(&process::Report) -> Option<Result<T, crate::epoch::Failure>>,
) -> Result<T, crate::epoch::Failure> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(unanswered(process, timeout, what));
        }
        let Some((engine, report)) = commands.report(remaining) else {
            return Err(unanswered(process, timeout, what));
        };
        if let Some(failure) = crate::epoch::Failure::of_report(&report) {
            update.fail(failure.clone());
            return Err(failure);
        }
        if engine != process.engine {
            tracing::debug!(
                engine,
                waiting_for = process.engine,
                "a main process reported while a switch waited"
            );
            continue;
        }
        if let Some(outcome) = matches(&report) {
            if let Err(failure) = &outcome {
                update.fail(failure.clone());
            }
            tracing::info!(
                host = %process.host,
                engine = process.engine,
                command = what,
                "a main process answered"
            );
            return outcome;
        }
    }
}

/// The failure of a main process that did not answer within its time.
fn unanswered(process: &EpochProcess, timeout: Duration, what: &str) -> crate::epoch::Failure {
    crate::epoch::Failure {
        kind: "timeout".to_owned(),
        reason: format!(
            "the main process of {} (engine {}) did not answer its {} within {} s",
            process.host,
            process.engine,
            what,
            timeout.as_secs()
        ),
        device: None,
    }
}

/// The outcome of a report that answers a command.
fn answered(report: &process::Report, matched: bool) -> Result<(), crate::epoch::Failure> {
    if matched {
        return Ok(());
    }
    Err(crate::epoch::Failure {
        kind: "out_of_order".to_owned(),
        reason: format!(
            "the main process answered a generation the switch did not ask for: {report:?}"
        ),
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
    use crate::engine::stand_in::{Answer, Seen, main_processes, process};
    use crate::options::ServeConfig;

    /// An epoch's start names this daemon's own build only for its own binary,
    /// and starts the process with the launch file alone: another executable's
    /// children fix their identity with their first `Hello` (§7.4).
    #[test]
    fn an_epoch_start_names_the_build_of_the_daemons_own_executable() {
        let directory = tempfile::tempdir().unwrap();
        let control = EngineControl::new(
            &directory.path().join("requests.sock"),
            "unix:/run/processes.sock".into(),
            "rank-0".into(),
            "model".into(),
            Some(std::path::PathBuf::from("/run/program-server.sock")),
        )
        .unwrap();
        control
            .write_build_config(3, &crate::epoch::BuildConfig::of(&ServeConfig::default()))
            .unwrap();
        let own = std::env::current_exe().unwrap().display().to_string();
        // The deployment's own spec, as `ServeOptions` builds it: an epoch's
        // start reads the ranks it places, and a spec with no scheduling or
        // cache tables is no deployment's.
        let mut spec = ServeConfig::default().engine_spec().unwrap();
        // The leader completes the options' spec with what the loaded model
        // determines, and a start reads it back (`crate::leader`), so a spec an
        // epoch could start always states facts. This test starts a process,
        // not a model, so it states the empty ones the crate's other specs do.
        spec.model_facts = Some(Default::default());
        spec.deployment = "model".to_owned();
        spec.model = "/models/qwen3".to_owned();
        spec.host = "rank-0".to_owned();
        spec.world = 3;
        let placement = spec
            .placement
            .as_mut()
            .expect("the default spec places ranks");
        placement.devices = vec![0];
        placement.queue_depth = 4;
        spec.workers
            .as_mut()
            .expect("the default spec names the workers")
            .executable = own;

        let start = control.epoch_start(&spec, 9).unwrap();
        assert_eq!(start.build, Some(proto::BUILD.to_owned()));
        // The file is the adapter's to write for the process it starts; the
        // start carries its contents, with this process's generation.
        assert_eq!(start.launch.engine, 9);
        assert_eq!(start.launch.world, 3);
        assert_eq!(start.launch.processes, "unix:/run/processes.sock");
        assert_eq!(start.launch.requests, control.request_address());
        assert_eq!(start.launch.queue_depth, 4);
        assert_eq!(start.launch.world_size, 1);

        // The launch states whether the child stands by (§2.5): a standby has
        // to know the role before it claims storage, and a serving child must
        // not read the file as one.
        assert!(!start.launch.standby);
        spec.predecessor = Some(wire::Predecessor {
            world: 3,
            segments: Vec::new(),
        });
        let standby = control.epoch_start(&spec, 10).unwrap();
        assert!(standby.launch.standby);
        assert_eq!(standby.launch.world, 3);
        spec.predecessor = None;

        spec.workers.as_mut().unwrap().executable = "/opt/uniserve-b".to_owned();
        let other = control.epoch_start(&spec, 9).unwrap();
        assert_eq!(other.build, None);
        assert_eq!(other.executable, "/opt/uniserve-b");
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
        let (processes, seen) =
            main_processes(vec![(7, Answer::Serves(11)), (8, Answer::Serves(12))]);
        let target = crate::epoch::Deployment::of(ServeConfig::default());
        let mut switch = ProgramSwitch::new(
            &processes,
            vec![process("rank-0", 7), process("rank-1", 8)],
            update(),
        );
        switch
            .precopy(&target.programs.kernels, &target.programs.policy)
            .expect("both main processes prepared the programs");
        switch
            .switchover()
            .expect("both switched to what they prepared");
        switch.commit().expect("both committed");
        switch.purge().expect("both purged what they no longer run");
        assert_eq!(switch.update().state, crate::epoch::UpdateState::Committed);

        // Each process's own number comes back to it: the daemon does not
        // invent one.
        let seen = seen.lock().unwrap().clone();
        assert_eq!(
            seen,
            vec![
                (7, Seen::Precopy),
                (8, Seen::Precopy),
                (7, Seen::Switchover(11)),
                (8, Seen::Switchover(12)),
                (7, Seen::Commit),
                (8, Seen::Commit),
                (7, Seen::Purge(11)),
                (8, Seen::Purge(12)),
            ]
        );
    }

    /// A main process that cannot prepare the programs fails the update and
    /// keeps serving the generation it runs: nothing is switched, and the update
    /// is not unavailable, since nothing reverted (§7.3 step 2).
    #[test]
    fn a_failed_precopy_ends_the_switch_before_any_generation_moves() {
        let (processes, seen) = main_processes(vec![
            (7, Answer::Serves(11)),
            (
                8,
                Answer::Refuses("the package has no module for this GPU".to_owned()),
            ),
        ]);
        let target = crate::epoch::Deployment::of(ServeConfig::default());
        let mut switch = ProgramSwitch::new(
            &processes,
            vec![process("rank-0", 7), process("rank-1", 8)],
            update(),
        );
        let failure = switch
            .precopy(&target.programs.kernels, &target.programs.policy)
            .expect_err("a refused precopy fails");
        assert_eq!(failure.kind, "precopy_failed");
        assert!(
            failure.reason.contains("no module for this GPU"),
            "{}",
            failure.reason
        );
        // Nothing moved: only the precopies were sent, and a switchover without
        // a prepared generation is refused rather than sent.
        let seen = seen.lock().unwrap().clone();
        assert!(
            seen.iter().all(|(_, what)| *what == Seen::Precopy),
            "{seen:?}"
        );
        assert!(switch.switchover().is_err());
        assert_eq!(switch.update().state, crate::epoch::UpdateState::Probation);
        assert!(switch.update().unavailable().is_none());
    }

    /// A main process's fault while the programs are prepared fails the update
    /// with that first failure, and a reverted update stays unavailable until
    /// `force` is asked for (§7.3 step 4).
    #[test]
    fn a_fault_during_the_precopy_is_the_updates_first_failure() {
        let (processes, _) = main_processes(vec![(
            7,
            Answer::Reports(|| process::Report {
                kind: Some(report::Kind::Fault(process::Fault {
                    kind: "sticky".to_owned(),
                    reason: "an earlier launch failed".to_owned(),
                    device: 1,
                })),
            }),
        )]);
        let target = crate::epoch::Deployment::of(ServeConfig::default());
        let mut switch = ProgramSwitch::new(&processes, vec![process("rank-0", 7)], update());
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

    /// A scheduling policy that fails reports `PolicyReverted` without being
    /// asked, and that report is the update's first failure (§4.4).
    #[test]
    fn a_policy_revert_while_the_programs_are_prepared_fails_the_update() {
        let (processes, _) = main_processes(vec![(
            7,
            Answer::Reports(|| process::Report {
                kind: Some(report::Kind::PolicyReverted(process::PolicyReverted {
                    scheduler: Some(0),
                    backend: "priority".to_owned(),
                    revision: 3,
                    reason: "the trial refused a decision".to_owned(),
                    fatal: false,
                })),
            }),
        )]);
        let target = crate::epoch::Deployment::of(ServeConfig::default());
        let mut switch = ProgramSwitch::new(&processes, vec![process("rank-0", 7)], update());
        let failure = switch
            .precopy(&target.programs.kernels, &target.programs.policy)
            .expect_err("a policy revert fails the update");
        assert_eq!(failure.kind, "policy_reverted");
        assert!(failure.reason.contains("priority"), "{}", failure.reason);
    }
}

/// The CUDA ordinals `host`'s ranks serve, in rank order and without repeats.
///
/// A deployment that states no `placement_config.devices` still gives each
/// host's ranks the ordinals the engine's shorthand numbers them from zero; one
/// that does state an order has already had it folded into the ranks by
/// [`engine::EngineConfig::from_spec`]. Reading the ranks is the one account of
/// both, and it is what the launch file states as the devices of this node's
/// process (§2.3, §7.4).
///
/// # Errors
///
/// Fails when the spec does not resolve to an engine configuration, when a rank
/// of `host` does not run on a CUDA device, or when no rank of the deployment
/// runs on `host`.
fn local_devices(host: &str, spec: &wire::EngineSpec) -> anyhow::Result<Vec<u32>> {
    let config = engine::EngineConfig::from_spec(spec)?;
    let mut devices = Vec::new();
    for group in &config.worker_groups {
        for rank in &group.ranks {
            if rank.host != host {
                continue;
            }
            let ordinal = rank
                .device
                .strip_prefix("cuda:")
                .with_context(|| {
                    format!(
                        "host {host} runs a rank on {:?}: a main process serves CUDA devices",
                        rank.device
                    )
                })?
                .parse::<u32>()
                .with_context(|| {
                    format!(
                        "the device ordinal of rank {:?} does not parse",
                        rank.device
                    )
                })?;
            if !devices.contains(&ordinal) {
                devices.push(ordinal);
            }
        }
    }
    anyhow::ensure!(
        !devices.is_empty(),
        "the placement names no device for this rank: no rank of the deployment runs on {host}"
    );
    Ok(devices)
}
