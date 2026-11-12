//! Starting the deployment's engine process and connecting to it.
//!
//! The leader daemon starts each engine process as a child of a node's
//! supervision, through the process interface like any process of the
//! deployment: [`EngineControl`] connects to the daemon's process socket and
//! asks it to spawn `uniserve engine` on the node, which connects back to the
//! daemon's request socket and says hello. The daemon answers with the
//! engine's specification and waits until the engine reports that it serves;
//! the engine's process lives for as long as its control connection does.
//!
//! An engine whose worker group has ranks on several hosts is one main process
//! per host ([`world_hosts`]): the scheduler node's, on the deployment's engine
//! host, and a peer on each other host. They are started together, the daemon
//! having reserved on the scheduler node the two sockets the peers dial (the
//! peer channel and the ranks' host-collective rendezvous), and the engine is
//! the scheduler node's client, which holds its peers' and ends with them.

use proto::config::EngineArgs;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use proto::process::{self, downstream, event, hello, reply, request, upstream};
use proto::request::wire;
use proto::{Connection, Listener};

use crate::engine::client::{EngineClient, EngineInfo, EngineState, StatsSink};

/// The worker group a daemon supervises an engine process under.
pub(crate) const ENGINE_GROUP: &str = "engine";

/// How long one read of a starting target's state waits. The start itself has
/// no bound: an engine loads its ranks, resolves its modules and captures its
/// graphs before it states that it parked, and one that exits meanwhile
/// closes this connection, which ends the wait.
const TARGET_STATE_READ: Duration = Duration::from_secs(60);
/// How long an engine process takes to connect back to the request socket
/// once the daemon started it.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// How long the daemon takes to answer the control connection.
const DAEMON_TIMEOUT: Duration = Duration::from_secs(90);

/// How often a wait rechecks the engine process while it starts.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Variables a main process's CUDA and NCCL libraries read before their first use, with the value it
/// runs with unless the daemon's own environment states the variable.
///
/// - `CUDA_MODULE_LOADING=EAGER`: lazy module loading synchronizes the context when a kernel is first
///   used, which can deadlock against a collective another stream is waiting in for a peer.
/// - `NCCL_RUNTIME_CONNECT=0`: deferred peer handshakes would allocate connection buffers after the
///   process booked its device storage.
const RUNTIME_DEFAULTS: [(&str, &str); 2] = [
    ("CUDA_MODULE_LOADING", "EAGER"),
    ("NCCL_RUNTIME_CONNECT", "0"),
];

/// The environment a main process is started under: stated on its command, so it precedes any CUDA
/// or NCCL initialization.
fn main_process_environment() -> std::collections::HashMap<String, String> {
    RUNTIME_DEFAULTS
        .into_iter()
        .map(|(name, value)| {
            let value = std::env::var(name).unwrap_or_else(|_| value.to_owned());
            (name.to_owned(), value)
        })
        .collect()
}

/// How long a Build Server this daemon starts may take to listen: it
/// discovers its toolchains, which imports the compilers of its Python
/// environment, and loads every installed package first.
const BUILD_SERVER_START: std::time::Duration = std::time::Duration::from_secs(120);

/// Makes sure this host's Build Server serves on the socket `settings`
/// names, and returns that socket: when none serves, the Build Server is
/// started with `settings` as a host service of this executable
/// (`service::ensure`), which outlives this daemon.
///
/// # Errors
///
/// Fails when no Build Server serves and none can be started.
pub fn ensure_build_server(settings: &build_server::command::Settings) -> anyhow::Result<PathBuf> {
    let socket = settings.socket();
    let executable = std::env::current_exe().context("locating this executable")?;
    ::service::ensure(
        &executable,
        "build-server",
        &socket,
        &settings.arguments(),
        BUILD_SERVER_START,
    )
    .context("starting this host's Build Server")?;
    Ok(socket)
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
    /// This node's Build Server socket, which every engine process is told
    /// to resolve an epoch's modules from (§5.3, §7.3).
    build_server: PathBuf,
    next_engine: AtomicU64,
    /// Engine handoffs can connect in any order. Reading and assigning
    /// their greetings is serialized; each client keeps its buffered reader.
    incoming: std::sync::Mutex<BTreeMap<u64, (Connection, Connection, wire::Hello)>>,
}

impl EngineControl {
    /// Binds the request socket at `socket` for the engines of `deployment`,
    /// which run on host `host` through the daemon whose process socket is
    /// `processes` and resolve their modules at the Build Server on
    /// `build_server` ([`ensure_build_server`]).
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be bound.
    pub fn new(
        socket: &Path,
        processes: String,
        host: String,
        deployment: String,
        build_server: PathBuf,
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
            build_server,
            next_engine: AtomicU64::new(1),
            incoming: std::sync::Mutex::new(BTreeMap::new()),
        })
    }

    /// Starts an engine process with `args` and returns its client once it reports that it
    /// serves, or, for arguments that name the process it succeeds ([`EngineArgs::standby`]), once
    /// it reports that it stands by; `stats` receives its schedulers' statistics. The process runs
    /// the executable the engine configuration names (`worker_config.executable`). This control
    /// states the process's place and interfaces in the arguments it writes ([`Self::launch`]).
    /// Blocks for the engine's whole start, its workers' loading included.
    ///
    /// # Errors
    ///
    /// Fails when the daemon cannot start the process, the process exits or
    /// does not connect back, its build speaks another major version, or it
    /// reports that its start failed.
    pub fn start(&self, args: EngineArgs, stats: StatsSink) -> anyhow::Result<EngineClient> {
        let engine = self.next_engine.fetch_add(1, Ordering::Relaxed);
        self.start_at(args, engine, stats)
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
        args: EngineArgs,
        engine: u64,
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
        self.next_engine.fetch_max(engine + 1, Ordering::Relaxed);
        let hosts = world_hosts(&args.engine_config)?;
        // The peers' numbers follow the scheduler node's from the same counter, so one number names
        // one process of the deployment.
        let numbers: Vec<u64> = std::iter::once(engine)
            .chain((1..hosts.len()).map(|_| self.next_engine.fetch_add(1, Ordering::Relaxed)))
            .collect();
        self.start_world(args, &numbers, stats)
    }

    /// Starts the engine of `args` as one main process per host of its worker group, numbered
    /// `numbers` in [`world_hosts`]'s order (the scheduler node's first), and returns the scheduler
    /// node's client, which holds the peers'. Each process reports that it serves, or that it
    /// stands by when `args` names a predecessor.
    ///
    /// # Errors
    ///
    /// Fails as [`EngineControl::start`] does, for any of the processes; a process that started is
    /// ended with its control connection.
    pub fn start_world(
        &self,
        args: EngineArgs,
        numbers: &[u64],
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
        let hosts = world_hosts(&args.engine_config)?;
        anyhow::ensure!(
            numbers.len() == hosts.len(),
            "{} engine numbers for the main processes of {} hosts",
            numbers.len(),
            hosts.len()
        );
        for number in numbers {
            self.next_engine.fetch_max(number + 1, Ordering::Relaxed);
        }
        let world_size = hosts.iter().map(|host| host.devices.len()).sum::<usize>();
        let world_size = u32::try_from(world_size).context("a world wider than u32 ranks")?;
        let interfaces = Interfaces {
            requests: &self.request_address,
            processes: &self.processes,
            build_server: &self.build_server,
        };
        let mut launches = hosts
            .iter()
            .zip(numbers)
            .map(|(host, number)| place_on(args.clone(), *number, &interfaces, host, world_size))
            .collect::<anyhow::Result<Vec<_>>>()?;
        // A new epoch takes its first engine's number, on every host.
        let epoch = launches[0].epoch;
        for launch in &mut launches {
            launch.epoch = epoch;
        }
        let mut inherit = vec![Vec::new(); launches.len()];
        if launches.len() > 1 {
            let (channel, store) = self.reserve_world(&hosts[0].host)?;
            let world = proto::config::World {
                scheduler: numbers[0],
                channel: channel.address.clone(),
                store: store.address.clone(),
                peers: u32::try_from(launches.len() - 1).unwrap_or(u32::MAX),
                inherited: None,
            };
            for launch in &mut launches {
                launch.world = Some(world.clone());
            }
            if let Some(scheduler) = launches[0].world.as_mut() {
                scheduler.inherited = Some(proto::config::Inherited {
                    channel: channel.descriptor,
                    store: store.descriptor,
                });
            }
            inherit[0] = vec![channel.reservation, store.reservation];
            tracing::info!(
                hosts = ?hosts.iter().map(|host| host.host.as_str()).collect::<Vec<_>>(),
                engines = ?numbers,
                channel = %world.channel,
                store = %world.store,
                "starting an engine whose worker group spans hosts"
            );
        }
        let parked = launches[0].parked();

        // Every process starts before any is awaited: the scheduler node's waits for its peers.
        let mut controls = Vec::with_capacity(launches.len());
        for (launch, inherit) in launches.iter().zip(inherit) {
            controls.push(self.spawn(launch, inherit)?);
        }
        let mut links = Vec::with_capacity(launches.len());
        for (launch, control) in launches.iter().zip(controls.iter_mut()) {
            links.push(self.welcome(launch.engine, control)?);
        }
        let started = await_started(&mut links, &mut controls, &launches, parked)?;

        let mut clients = Vec::with_capacity(launches.len());
        let mut sinks = std::iter::once(stats);
        for (((reader, writer), control), (launch, (info, state))) in links
            .into_iter()
            .zip(controls)
            .zip(launches.iter().zip(started))
        {
            // A peer holds no scheduler, so it reports no statistics.
            let sink = sinks.next().unwrap_or_else(|| Box::new(|_, _, _| {}));
            clients.push(
                EngineClient::new(reader, writer, info, state, sink, Some(control))?
                    .numbered(launch.engine)
                    .in_epoch(epoch)
                    .on_host(&launch.host),
            );
        }
        let mut clients = clients.into_iter();
        let scheduler = clients.next().context("an engine of no main process")?;
        Ok(scheduler.with_peers(clients.collect()))
    }

    /// Reserves on `host` the world's two sockets, which the scheduler node's process inherits:
    /// its peer channel's listener and its ranks' host-collective rendezvous.
    fn reserve_world(&self, host: &str) -> anyhow::Result<(process::Reserved, process::Reserved)> {
        let (mut control, _) = self.connect()?;
        let mut reserve = |id| -> anyhow::Result<process::Reserved> {
            control.send(&process::Upstream {
                kind: Some(upstream::Kind::Request(process::Request {
                    id,
                    kind: Some(request::Kind::Reserve(process::Reserve {
                        host: host.to_owned(),
                        group: ENGINE_GROUP.to_owned(),
                        engine: 0,
                        routable: true,
                    })),
                })),
            })?;
            let reply = await_reply(&mut control, id, "a reservation")?;
            anyhow::ensure!(
                reply.error.is_empty(),
                "reserving a socket on {host}: {}",
                reply.error
            );
            match reply.kind {
                Some(reply::Kind::Reserved(reserved)) => Ok(reserved),
                _ => anyhow::bail!("the daemon answered a reservation with another reply"),
            }
        };
        let channel = reserve(1)?;
        let store = reserve(2)?;
        Ok((channel, store))
    }

    /// Accepts engine `engine`'s request link and welcomes it, or refuses it when it speaks
    /// another major version or states another number.
    fn welcome(
        &self,
        engine: u64,
        control: &mut Connection,
    ) -> anyhow::Result<(Connection, Connection)> {
        let (reader, mut writer, hello) = self.accept(engine, Some(control))?;
        let refusal = if hello.major != proto::ENGINE_INTERFACE_VERSION {
            format!(
                "the engine speaks major version {} and this daemon {}",
                hello.major,
                proto::ENGINE_INTERFACE_VERSION
            )
        } else if hello.engine != engine {
            format!("engine {} connected for engine {engine}", hello.engine)
        } else {
            String::new()
        };
        writer.send(&wire::Welcome {
            major: proto::ENGINE_INTERFACE_VERSION,
            build: proto::BUILD.to_owned(),
            refusal: refusal.clone(),
        })?;
        anyhow::ensure!(refusal.is_empty(), "refused the engine process: {refusal}");
        tracing::info!(engine, build = %hello.build, "the engine process connected; awaiting its start");
        Ok((reader, writer))
    }

    /// The request socket's address, which an engine attaches to
    /// (`EngineClient::attach`).
    /// The host this control's engines run on.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// This node's Build Server socket.
    pub fn build_server(&self) -> &std::path::Path {
        &self.build_server
    }

    pub fn request_address(&self) -> &str {
        &self.request_address
    }

    /// The identity the next engine this control starts takes.
    pub fn next_engine(&self) -> u64 {
        self.next_engine.load(Ordering::Relaxed)
    }

    /// Joins an epoch update's target, which executes `epoch`: the engine the update started
    /// dials this daemon's request socket, is welcomed, and states that it parked. The update
    /// commands it on the process link, so this connection carries only its requests once it
    /// serves.
    ///
    /// # Errors
    ///
    /// Fails when the engine does not connect in time, speaks another major version, fails to
    /// start, or exits before it stated that it parked.
    pub fn join_target(
        &self,
        engine: u64,
        epoch: u64,
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
        let (mut reader, mut writer, hello) = self.accept(engine, None)?;
        let refusal = if hello.major != proto::ENGINE_INTERFACE_VERSION {
            format!(
                "the engine speaks major version {} and this daemon {}",
                hello.major,
                proto::ENGINE_INTERFACE_VERSION
            )
        } else {
            String::new()
        };
        writer.send(&wire::Welcome {
            major: proto::ENGINE_INTERFACE_VERSION,
            build: proto::BUILD.to_owned(),
            refusal: refusal.clone(),
        })?;
        anyhow::ensure!(refusal.is_empty(), "refused the engine: {refusal}");
        let started = Instant::now();
        loop {
            let Some(message) = reader.recv::<wire::FromEngine>(TARGET_STATE_READ)? else {
                tracing::info!(
                    engine,
                    waited_s = started.elapsed().as_secs(),
                    "the update's target is still starting"
                );
                continue;
            };
            let Some(wire::from_engine::Kind::State(state)) = message.kind else {
                continue;
            };
            let known = match wire::EngineState::try_from(state.state) {
                Ok(wire::EngineState::Parked) => EngineState::Parked,
                Ok(wire::EngineState::Serving) => EngineState::Serving,
                Ok(wire::EngineState::Failed) => {
                    anyhow::bail!("the target failed to start: {}", state.error)
                }
                _ => continue,
            };
            let info = state.info.context("the engine reported no info")?;
            tracing::info!(engine, build = %hello.build, "the update's target joined this daemon");
            return EngineClient::new(reader, writer, info.into(), known, stats, None)
                .map(|client| client.numbered(engine).in_epoch(epoch));
        }
    }

    /// Takes over the engine `engine` a daemon this one replaced started,
    /// which executes `epoch` and lives under `control`: the engine connects
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
        epoch: u64,
        mut control: Option<Connection>,
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
        self.next_engine.fetch_max(next_engine, Ordering::Relaxed);
        let (mut reader, mut writer, hello) = self.accept(engine, control.as_mut())?;
        let refusal = if hello.major != proto::ENGINE_INTERFACE_VERSION {
            format!(
                "the engine speaks major version {} and this daemon {}",
                hello.major,
                proto::ENGINE_INTERFACE_VERSION
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
            major: proto::ENGINE_INTERFACE_VERSION,
            build: proto::BUILD.to_owned(),
            refusal: refusal.clone(),
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
                    state.state == wire::EngineState::Serving as i32,
                    "engine {engine} attached while it does not serve: {}",
                    state.error
                );
                let info = state.info.context("the engine reported no info")?;
                tracing::info!(engine, build = %hello.build, "the engine attached");
                return EngineClient::new(
                    reader,
                    writer,
                    info.into(),
                    EngineState::Serving,
                    stats,
                    control,
                )
                .map(|client| client.numbered(engine).in_epoch(epoch));
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
            major: proto::ENGINE_INTERFACE_VERSION,
            build: proto::BUILD.to_owned(),
            peer: Some(hello::Peer::Engine(process::Engine {
                deployment: self.deployment.clone(),
                // This connection asks the daemon's supervisor to start and stop processes; it is
                // not one itself, so it states no engine number and no interface, and no command is
                // addressed to it.
                engine: 0,
                interface: 0,
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

    /// The storage ledgers every node's Storage Server keeps for the deployment, one JSON report
    /// per node: `{"host": ..., "ledgers": [...], "host_storage": {...}, "transfers": [...]}`.
    /// Each node's daemon reads them over its claim, so no main process is asked.
    pub(crate) fn storage_reports(&self) -> anyhow::Result<Vec<String>> {
        let (mut connection, nodes) = self.connect()?;
        let mut reports = Vec::with_capacity(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            let host = node.host.clone();
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
                .context("the daemon did not report its storage ledgers")?;
            let Some(downstream::Kind::Reply(answer)) = answer.kind else {
                anyhow::bail!("the daemon did not answer its ledger query");
            };
            anyhow::ensure!(
                answer.error.is_empty(),
                "the storage ledgers of {host}: {}",
                answer.error
            );
            let Some(reply::Kind::Ledger(report)) = answer.kind else {
                anyhow::bail!("the daemon returned no storage ledger");
            };
            let mut report: serde_json::Value = serde_json::from_str(&report.ledgers)?;
            report["host"] = serde_json::Value::String(host);
            reports.push(report.to_string());
        }
        Ok(reports)
    }

    /// Publishes one commit decision at every deployment node. Publication
    /// is atomic within each Storage Server and idempotent across retries;
    /// the daemon retains its commit decision until every node has answered.
    pub fn publish_epochs(
        &self,
        epochs: &[u64],
        pending: &mut Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let (mut connection, nodes) = self.connect()?;
        let pending =
            pending.get_or_insert_with(|| nodes.into_iter().map(|node| node.host).collect());
        while let Some(host) = pending.first().cloned() {
            connection.send(&process::Upstream {
                kind: Some(upstream::Kind::Request(process::Request {
                    id: 1,
                    kind: Some(request::Kind::PublishEpochs(process::PublishEpochs {
                        host: host.clone(),
                        deployment: self.deployment.clone(),
                        epochs: epochs.to_vec(),
                    })),
                })),
            })?;
            let answer: process::Downstream = connection
                .recv(DAEMON_TIMEOUT)?
                .context("the daemon did not report epoch publication")?;
            let Some(downstream::Kind::Reply(answer)) = answer.kind else {
                anyhow::bail!("the daemon did not answer epoch publication");
            };
            anyhow::ensure!(
                answer.error.is_empty(),
                "publishing epochs at {host}: {}",
                answer.error
            );
            anyhow::ensure!(
                matches!(answer.kind, Some(reply::Kind::EpochsPublished(_))),
                "the daemon returned no publication result"
            );
            pending.remove(0);
        }
        Ok(())
    }

    /// Has every node's daemon halve, or restore when not `halve`, the byte
    /// budgets of the links the deployment's transfers use, and answers the
    /// hosts that took it. A node that holds no claim of the deployment has
    /// nothing to throttle and says so by answering.
    ///
    /// # Errors
    ///
    /// Fails when a node's daemon cannot be reached or refuses.
    pub(crate) fn throttle_transfers(&self, halve: bool) -> anyhow::Result<Vec<String>> {
        let (mut connection, nodes) = self.connect()?;
        let mut hosts = Vec::with_capacity(nodes.len());
        for node in nodes {
            connection.send(&process::Upstream {
                kind: Some(upstream::Kind::Request(process::Request {
                    id: 1,
                    kind: Some(request::Kind::ThrottleTransfers(
                        process::ThrottleTransfers {
                            host: node.host.clone(),
                            deployment: self.deployment.clone(),
                            halve,
                        },
                    )),
                })),
            })?;
            let answer: process::Downstream = connection
                .recv(DAEMON_TIMEOUT)?
                .context("the daemon did not report the throttle")?;
            let Some(downstream::Kind::Reply(answer)) = answer.kind else {
                anyhow::bail!("the daemon did not answer the throttle");
            };
            anyhow::ensure!(
                answer.error.is_empty(),
                "throttling transfers at {}: {}",
                node.host,
                answer.error
            );
            hosts.push(node.host);
        }
        Ok(hosts)
    }

    /// The launch file's contents for the scheduler node's process `engine`: `args`, with this
    /// control's statement of the process's place and interfaces ([`place`]).
    ///
    /// # Errors
    ///
    /// As [`place`].
    fn launch(&self, args: EngineArgs, engine: u64) -> anyhow::Result<EngineArgs> {
        place(
            args,
            engine,
            &Interfaces {
                requests: &self.request_address,
                processes: &self.processes,
                build_server: &self.build_server,
            },
        )
    }

    /// What an epoch update starts its target engine with: the launch file's contents for the
    /// engine of launch generation `engine`, and the variables set over the daemon's environment.
    ///
    /// # Errors
    ///
    /// Fails for a worker group with ranks on several hosts, which needs one launch per host,
    /// and as [`place`] does.
    pub(crate) fn target_launch(
        &self,
        args: &EngineArgs,
        engine: u64,
    ) -> anyhow::Result<(EngineArgs, std::collections::HashMap<String, String>)> {
        anyhow::ensure!(
            world_hosts(&args.engine_config)?.len() == 1,
            "an epoch update starts its target engines from one launch, and a worker group with \
             ranks on several hosts starts one per host: such a deployment changes epochs by a \
             relaunch"
        );
        Ok((
            self.launch(args.clone(), engine)?,
            main_process_environment(),
        ))
    }

    /// Has the daemon of `launch`'s host start the main process `launch` describes, inheriting the
    /// sockets of `inherit`; returns the control connection the process lives under.
    fn spawn(&self, launch: &EngineArgs, inherit: Vec<u64>) -> anyhow::Result<Connection> {
        let engine = launch.engine;
        let host = launch.host.as_str();
        let (mut control, nodes) = self.connect()?;
        anyhow::ensure!(
            nodes.iter().any(|node| node.host == host),
            "engine host {host} has not joined the deployment"
        );
        anyhow::ensure!(
            host == self.host || !self.processes.starts_with("unix:"),
            "a remote engine needs the deployment's network control address"
        );

        control.send(&process::Upstream {
            kind: Some(upstream::Kind::Request(process::Request {
                id: 1,
                kind: Some(request::Kind::Spawn(process::Spawn {
                    host: host.to_owned(),
                    group: ENGINE_GROUP.to_owned(),
                    rank: launch.rank,
                    generation: engine,
                    // The node that starts the process resolves the engine
                    // package its launch names at its own Build Server.
                    executable: String::new(),
                    arguments: Vec::new(),
                    environment: main_process_environment(),
                    inherit,
                    // The node that starts the process writes its launch file.
                    launch: serde_json::to_vec(launch).context("encoding the launch file")?,
                })),
            })),
        })?;
        let answer = await_reply(&mut control, 1, "the engine's start")?;
        anyhow::ensure!(
            answer.error.is_empty(),
            "starting the engine process on {host}: {}",
            answer.error
        );
        if let Some(reply::Kind::Spawned(spawned)) = answer.kind {
            tracing::info!(
                engine,
                host,
                pid = spawned.pid,
                "started the engine process"
            );
        }
        Ok(control)
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
        // Whether the handed-over control connection has been seen to close: the service
        // goes on with its accepted side, so an end here is a predecessor that left without
        // handing it over, and not the engine. Stated once.
        let mut control_closed = false;
        loop {
            match self.requests.try_accept() {
                Ok(Some(mut reader)) => {
                    let writer = reader.writer()?;
                    let hello: wire::Hello = reader
                        .recv(CONNECT_TIMEOUT)?
                        .context("the engine said no hello")?;
                    // Whose dial this is: a successor waits for one engine's front, and a
                    // connection it cannot use is otherwise indistinguishable from one that never
                    // arrived — the difference between an engine that dialled the wrong daemon and
                    // one that never dialled at all.
                    tracing::info!(
                        awaited = engine,
                        stated = hello.engine,
                        interface = hello.interface,
                        "an engine connected to this daemon's request socket"
                    );
                    if hello.engine == engine {
                        return Ok((reader, writer, hello));
                    }
                    incoming.insert(hello.engine, (reader, writer, hello));
                }
                Ok(None) => {}
                Err(error) => return Err(error).context("accepting the engine's connection"),
            }
            if let Some(control) = control.as_deref_mut()
                && !control_closed
            {
                match still_serving(control) {
                    Ok(true) => {}
                    // The handover's own connection, whose accepted side was not handed over:
                    // this end reads closed at the first look. Which peer ended is the whole
                    // content of the failure, and this one is the predecessor - the engine may
                    // be serving perfectly well, and the dial below is what says so.
                    Ok(false) => {
                        control_closed = true;
                        tracing::info!(
                            engine,
                            "the handed-over control connection closed; waiting for the engine \
                             to dial this daemon's request socket"
                        );
                    }
                    Err(error) => {
                        tracing::warn!(
                            engine,
                            %error,
                            "the engine's process connection closed while this daemon was adopting it"
                        );
                        return Err(error);
                    }
                }
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

/// Reads the reply `id` on a control connection, within the daemon's answer time.
fn await_reply(control: &mut Connection, id: u64, what: &str) -> anyhow::Result<process::Reply> {
    let deadline = Instant::now() + DAEMON_TIMEOUT;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        anyhow::ensure!(
            !remaining.is_zero(),
            "the daemon did not answer {what} in time"
        );
        let Some(message) = control.recv::<process::Downstream>(remaining)? else {
            continue;
        };
        if let Some(downstream::Kind::Reply(answer)) = message.kind
            && answer.id == id
        {
            return Ok(answer);
        }
    }
}

/// Waits until every process of a world reported that it serves, or that it stands by when
/// `standby`, and answers what each reported. The first failure ends the wait: a process that
/// reports its start failed, or exits, as its control connection says.
fn await_started(
    links: &mut [(Connection, Connection)],
    controls: &mut [Connection],
    launches: &[EngineArgs],
    parked: bool,
) -> anyhow::Result<Vec<(EngineInfo, EngineState)>> {
    let mut started: Vec<Option<(EngineInfo, EngineState)>> =
        (0..links.len()).map(|_| None).collect();
    loop {
        for (index, (reader, _)) in links.iter_mut().enumerate() {
            if started[index].is_some() {
                continue;
            }
            let engine = launches[index].engine;
            let host = &launches[index].host;
            while let Some(message) = reader.try_recv::<wire::FromEngine>()? {
                let Some(wire::from_engine::Kind::State(state)) = message.kind else {
                    continue;
                };
                match wire::EngineState::try_from(state.state) {
                    Ok(reported @ (wire::EngineState::Serving | wire::EngineState::Parked))
                        if (reported == wire::EngineState::Parked) == parked =>
                    {
                        let info = state.info.context("the engine reported no info")?;
                        let state = if parked {
                            tracing::info!(
                                engine,
                                host,
                                "the engine parked beside its predecessor"
                            );
                            EngineState::Parked
                        } else {
                            tracing::info!(
                                engine,
                                host,
                                schedulers = info.schedulers,
                                "the engine serves"
                            );
                            EngineState::Serving
                        };
                        started[index] = Some((info.into(), state));
                        break;
                    }
                    Ok(wire::EngineState::Failed) => {
                        // The failed engine closes its workers and exits;
                        // their storage is free once it has.
                        await_exit(reader, &mut controls[index], FAILED_EXIT);
                        anyhow::bail!("the engine on {host} failed to start: {}", state.error);
                    }
                    _ => {}
                }
            }
        }
        if started.iter().all(Option::is_some) {
            return Ok(started.into_iter().flatten().collect());
        }
        for (index, control) in controls.iter_mut().enumerate() {
            if started[index].is_none() {
                exited(control)
                    .with_context(|| format!("the engine on {}", launches[index].host))?;
            }
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// How long a failed engine takes to close its workers and exit, at most.
const FAILED_EXIT: Duration = Duration::from_secs(120);

/// Whether the control connection still reports a serving engine: `Ok(false)` when it has
/// closed, which is a connection ending rather than an engine exit.
fn still_serving(control: &mut Connection) -> anyhow::Result<bool> {
    match exited(control) {
        Ok(()) => Ok(true),
        Err(error) => match error.downcast_ref::<proto::Error>() {
            Some(proto::Error::Closed) => Ok(false),
            _ => Err(error),
        },
    }
}

/// Waits up to `timeout` for the daemon to report the engine process's exit
/// on its control connection; a wait that times out is logged.
fn await_exit(reader: &mut Connection, control: &mut Connection, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if exited(control).is_err() {
            return;
        }
        // A process that goes closes this connection, and the daemon then forgets it without
        // reporting an exit on `control`.
        match reader.try_recv::<wire::FromEngine>() {
            Ok(Some(_)) => {}
            Ok(None) if reader.wait(POLL_INTERVAL).is_ok() => {}
            _ => return,
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
            anyhow::bail!(
                "the engine process exited: {}: {}",
                exit.status,
                exit.last_line.trim()
            );
        }
    }
    Ok(())
}

/// One host's Build Server and the targets that host's devices build for
/// (§5.1, §5.3, §6.3).
#[derive(Debug, Clone)]
pub struct BuildServer {
    /// The host's identity in the deployment.
    pub host: String,
    /// Its Build Server's socket.
    pub socket: PathBuf,
    /// The targets its devices compile for, which `prepare` builds the
    /// epoch's modules for.
    pub targets: Vec<build_server::Target>,
}

/// Selects the modules `filled` names at every host's Build Server
/// (§5.1, §6.3):
///
/// The leader host selects each slot's backend at its latest installed
/// revision (`Selector::Latest`), which is the revision the epoch record
/// pins. Every other host is then asked for exactly those modules
/// (`Selector::Fixed`): the same revision names the same module on every
/// host, so a follower that lacks it fails the call rather than choosing
/// another.
///
/// The answer is `filled` with each package's revision and the parameters
/// the Build Server selected for it, which is what the epoch record keeps
/// and what `Precopy` sends to the main processes.
///
/// The step of the Build Server exchange the daemon is in.
///
/// The exchange waits twice in a way this crate cannot bound: the client opens
/// its socket, and reads the answer a Build Server may never send, with no
/// deadline (`crates/build-server/src/protocol.rs`: `Connection::connect` and
/// `Connection::receive`). Whoever times the call out reads the step recorded
/// here, so a deadline names the wait rather than the phase (§7.3 step 1).
#[derive(Clone)]
pub struct PrepareSteps(std::sync::Arc<std::sync::Mutex<PrepareStep>>);

#[derive(Clone)]
struct PrepareStep {
    host: String,
    what: &'static str,
}

impl Default for PrepareSteps {
    fn default() -> Self {
        Self::new()
    }
}

impl PrepareSteps {
    /// A record of an exchange that has not reached a Build Server yet.
    pub fn new() -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(PrepareStep {
            host: String::new(),
            what: "about to reach its first Build Server",
        })))
    }

    /// Records that the exchange with `host` is waiting at `what`.
    fn at(&self, host: &str, what: &'static str) {
        let mut step = self.0.lock().unwrap_or_else(|broken| broken.into_inner());
        step.host = host.to_owned();
        step.what = what;
    }

    /// The wait the daemon was in, and the host it was in it with.
    pub fn describe(&self) -> String {
        let step = self.0.lock().unwrap_or_else(|broken| broken.into_inner());
        if step.host.is_empty() {
            step.what.to_owned()
        } else {
            format!("{} ({})", step.what, step.host)
        }
    }
}

/// # Errors
///
/// Fails when a host has no Build Server in `servers`, when a Build
/// Server refuses or cannot build a module, and when a follower answers a
/// package other than the one the leader pinned.
pub fn prepare_epoch(
    servers: &[BuildServer],
    leader: &str,
    filled: &crate::epoch::ModuleSet,
    descriptors: &[build_server::Descriptors],
    steps: &PrepareSteps,
) -> anyhow::Result<crate::epoch::ModuleSet> {
    let leader_server = servers
        .iter()
        .find(|server| server.host == leader)
        .with_context(|| format!("no Build Server for the leader host {leader}"))?;
    tracing::info!(
        leader = %leader,
        servers = servers.len(),
        kernels = filled.kernels.len(),
        policy = %filled.policy.backend,
        "resolving the epoch's modules at its Build Servers"
    );
    let pinned = prepare_at(leader_server, filled, descriptors, steps)?;
    for server in servers {
        if server.host == leader {
            continue;
        }
        let selected = prepare_at(server, filled, descriptors, steps)?;
        anyhow::ensure!(
            selected == pinned,
            "the Build Server of {} selected other modules than the leader's",
            server.host
        );
    }
    tracing::info!(
        kernels = pinned.kernels.len(),
        policy = %pinned.policy.backend,
        "the epoch's modules are resolved"
    );
    Ok(pinned)
}

/// `filled` as a fixed module set: every package it names at the revision the leader's Build
/// Server selects for it, and every installed kernel package of every operator at its selected
/// revision.
///
/// This is the one selection of an apply. The daemon asks `cover` for each interface the
/// deployment uses: the engine, the scheduling policy and every kernel operator of this build.
/// A package the deployment names takes the revision it states, else its latest installed one.
/// Every candidate of an operator is recorded as well, one slot per package with the package as
/// its part: for each descriptor key the main process later asks `cover` to select among exactly
/// these revisions, so a package or a revision installed after this call is run by nobody until
/// a configuration is applied again, and a later revision of one of them is a package change a
/// configuration comparison sees.
///
/// # Errors
///
/// Fails when the leader's Build Server cannot be reached, or holds no package, or not the
/// revision, the deployment names for an interface.
pub fn with_store_modules(
    servers: &[BuildServer],
    leader: &str,
    filled: &crate::epoch::ModuleSet,
) -> anyhow::Result<crate::epoch::ModuleSet> {
    let leader_server = servers
        .iter()
        .find(|server| server.host == leader)
        .with_context(|| format!("no Build Server for the leader host {leader}"))?;
    let mut client = build_server::Client::connect(&leader_server.socket).with_context(|| {
        format!(
            "connecting to the Build Server of {} at {}",
            leader_server.host,
            leader_server.socket.display()
        )
    })?;

    let mut complete = filled.clone();
    fix_revision(&mut client, proto::module::ENGINE, &mut complete.engine)?;
    fix_revision(&mut client, POLICY_SLOT, &mut complete.policy)?;
    for (slot, package) in &mut complete.kernels {
        fix_revision(&mut client, &slot.op, package)?;
    }
    for operator in proto::module::operator::ALL {
        for candidate in candidates(&mut client, operator, &[])? {
            complete.kernels.insert(
                proto::module::ModuleSlot {
                    op: candidate.op,
                    phase: None,
                    part: Some(candidate.backend.clone()),
                },
                proto::module::PackageRevision {
                    backend: candidate.backend,
                    revision: candidate.revision,
                    arguments: toml::Table::new(),
                },
            );
        }
    }
    Ok(complete)
}

/// The candidates the Build Server selects among for `op`, given what the deployment prefers:
/// every installed package of the interface, each at its latest revision or the one `preferred`
/// states.
fn candidates(
    client: &mut build_server::Client,
    op: &str,
    preferred: &[build_server::Preference],
) -> anyhow::Result<Vec<build_server::protocol::Package>> {
    let coverage = client
        .cover(&build_server::CoverRequest {
            op: op.to_owned(),
            // No target: a module set fixes revisions for every host of the deployment, and each
            // main process selects among them for its own devices.
            targets: Vec::new(),
            keys: Vec::new(),
            preferred: preferred.to_vec(),
            fixed: Vec::new(),
        })
        .with_context(|| format!("asking the Build Server for the packages of {op}"))?;
    Ok(coverage.candidates)
}

/// Fixes the revision of the package `package` names for `op`: the one it states, checked, or
/// its latest installed one.
fn fix_revision(
    client: &mut build_server::Client,
    op: &str,
    package: &mut proto::module::PackageRevision,
) -> anyhow::Result<()> {
    let stated = build_server::Preference {
        phase: None,
        backend: package.backend.clone(),
        revision: (package.revision != 0).then_some(package.revision),
    };
    let selected = candidates(client, op, &[stated])?
        .into_iter()
        .find(|candidate| candidate.backend == package.backend)
        .with_context(|| format!("no {op} package {} is installed", package.backend))?;
    package.revision = selected.revision;
    Ok(())
}

/// Runs `prepare` at one host's Build Server for the module set `filled`,
/// whose revisions are fixed, and answers it with the parameters each
/// package runs with: its manifest's defaults under the deployment's
/// arguments.
fn prepare_at(
    server: &BuildServer,
    filled: &crate::epoch::ModuleSet,
    descriptors: &[build_server::Descriptors],
    steps: &PrepareSteps,
) -> anyhow::Result<crate::epoch::ModuleSet> {
    let policy_slot = proto::module::ModuleSlot::all(POLICY_SLOT);
    let engine_slot = proto::module::ModuleSlot::all(proto::module::ENGINE);
    let mut requests = Vec::with_capacity(filled.kernels.len() + 2);
    for (slot, package) in &filled.kernels {
        requests.push(build_server::PackageRequest {
            slot: slot.clone(),
            selector: build_server::Selector::Fixed(package.clone()),
        });
    }
    requests.push(build_server::PackageRequest {
        slot: policy_slot.clone(),
        selector: build_server::Selector::Fixed(filled.policy.clone()),
    });
    requests.push(build_server::PackageRequest {
        slot: engine_slot.clone(),
        selector: build_server::Selector::Fixed(filled.engine.clone()),
    });
    // The engine's one module is built here, on every host, before any node
    // starts a main process from it: a host whose store misses it builds the
    // executable for its own platform now, while nothing waits on a start.
    let mut descriptors = descriptors.to_vec();
    descriptors.push(build_server::Descriptors {
        slot: engine_slot.clone(),
        keys: vec![proto::module::SINGLE_KEY.to_owned()],
    });
    let request = build_server::PrepareRequest {
        targets: server.targets.clone(),
        packages: requests,
        descriptors,
    };
    // Before the connect, not after it: a daemon that blocks opening the
    // socket names the host it was reaching from this line alone, and the
    // server it never reached logs nothing at all.
    steps.at(&server.host, "opening its connection to a Build Server");
    tracing::info!(
        host = %server.host,
        socket = %server.socket.display(),
        "connecting to a Build Server"
    );
    let mut client = build_server::Client::connect(&server.socket).with_context(|| {
        format!(
            "connecting to the Build Server of {} at {}",
            server.host,
            server.socket.display()
        )
    })?;
    tracing::info!(
        host = %server.host,
        socket = %server.socket.display(),
        targets = ?server.targets,
        modules = request.packages.len(),
        "asking a Build Server to prepare the epoch's modules"
    );
    // A cold prepare builds every module it was asked for, which on this host
    // is minutes per package; the server reports each step as it goes. At
    // `debug` those steps were invisible in a deployment's log, so a prepare
    // that was working looked exactly like one that had died. The three states
    // that change what a reader waits for are at `info`, and `Queued` — one
    // line per module before anything happens — stays at `debug`.
    let mut progress = |step: build_server::Progress| {
        let building = matches!(
            step.state,
            build_server::BuildState::Building
                | build_server::BuildState::Committed
                | build_server::BuildState::Failed(_)
        );
        if building {
            tracing::info!(
                host = %server.host,
                module = %step.path,
                state = ?step.state,
                "preparing a package's module"
            );
        } else {
            tracing::debug!(
                host = %server.host,
                module = %step.path,
                state = ?step.state,
                "a package's module is queued"
            );
        }
    };
    steps.at(
        &server.host,
        "waiting for a Build Server's answer to the prepare",
    );
    let prepared = client
        .prepare(&request, &mut progress)
        .with_context(|| format!("preparing the epoch's modules at {}", server.host))?;
    steps.at(&server.host, "reading the modules a Build Server selected");
    tracing::info!(
        host = %server.host,
        prepared = prepared.len(),
        "a Build Server answered"
    );
    let mut kernels = BTreeMap::new();
    let mut policy = None;
    let mut engine = None;
    for (slot, package) in prepared {
        if slot == policy_slot {
            policy = Some(package);
        } else if slot == engine_slot {
            engine = Some(package);
        } else {
            kernels.insert(slot, package);
        }
    }
    let policy = policy.context("the Build Server selected no scheduling policy package")?;
    let engine = engine.context("the Build Server selected no engine package")?;
    anyhow::ensure!(
        kernels.len() == filled.kernels.len(),
        "the Build Server of {} selected modules for {} of the {} slots asked",
        server.host,
        kernels.len(),
        filled.kernels.len()
    );
    Ok(crate::epoch::ModuleSet {
        engine,
        kernels,
        policy,
    })
}

/// The slot the scheduling policy package fills: the name the policy packages state
/// as their `op` (`crates/scheduling/policies/<backend>/package.toml`). The
/// scheduling interface is not an operator (§5.1): the epoch record keeps its
/// package beside the kernels' rather than among them, so `prepare` names it
/// by this slot and the daemon takes it back out.
const POLICY_SLOT: &str = "scheduling";

/// One engine process on a node, by the engine number its process link is
/// registered under, which a command names and a report carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpochProcess {
    /// The node the main process runs on.
    pub host: String,
    /// The engine number its process connection is registered under, which a
    /// command names and a report carries.
    pub engine: u64,
}

impl crate::interference::Throttle for EngineControl {
    fn throttle(&self, halve: bool) -> Result<Vec<String>, String> {
        self.throttle_transfers(halve)
            .map_err(|error| format!("{error:#}"))
    }
}

/// What a node's main process reaches, as its launch file states it.
pub(crate) struct Interfaces<'a> {
    /// The leader daemon's request interface.
    pub(crate) requests: &'a str,
    /// The daemon's process interface.
    pub(crate) processes: &'a str,
    /// The Build Server's socket.
    pub(crate) build_server: &'a Path,
}

/// The launch file's contents for the scheduler node's process `engine`: `args`, with the
/// process's place in the deployment (its devices, group, first rank and world) and the interfaces
/// `interfaces` states. The scheduler node is the deployment's engine host.
///
/// A value the file requires and the daemon cannot state is an error here
/// rather than a default: the engine refuses a launch file that lacks one,
/// and a default would be a fact the daemon invented about the process.
///
/// # Errors
///
/// Fails for a configuration of no model, ranks the engine host does not lead
/// ([`world_hosts`]), a rank on a device that is not CUDA's, and a queue depth
/// the file cannot hold.
pub(crate) fn place(
    args: EngineArgs,
    engine: u64,
    interfaces: &Interfaces<'_>,
) -> anyhow::Result<EngineArgs> {
    let hosts = world_hosts(&args.engine_config)?;
    let world_size = hosts.iter().map(|host| host.devices.len()).sum::<usize>();
    place_on(
        args,
        engine,
        interfaces,
        &hosts[0],
        u32::try_from(world_size).context("a world wider than u32 ranks")?,
    )
}

/// The ranks of the deployment's worker groups one host holds: a main process's place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HostRanks {
    /// The host, by host identity.
    pub(crate) host: String,
    /// The group rank of its first rank; its others follow it.
    pub(crate) first: u32,
    /// The CUDA ordinals of its ranks, in rank order.
    pub(crate) devices: Vec<u32>,
}

/// The hosts of the deployment's worker groups, the engine host first, each with the consecutive
/// ranks it holds: one main process per host serves an epoch (`docs/architecture.md`, The
/// channel), the engine host's holding rank 0 and the schedulers.
///
/// # Errors
///
/// Fails when the engine host holds no rank or not rank 0, when a host's ranks are not
/// consecutive, or as [`local_devices`] fails.
pub(crate) fn world_hosts(config: &proto::config::EngineConfig) -> anyhow::Result<Vec<HostRanks>> {
    let engine_host = &config.parallel.engine_host;
    let mut order: Vec<String> = Vec::new();
    let mut previous: Option<&str> = None;
    for rank in config
        .parallel
        .worker_groups
        .iter()
        .flat_map(|group| group.ranks.iter())
    {
        if previous != Some(rank.host.as_str()) {
            anyhow::ensure!(
                !order.contains(&rank.host),
                "the ranks of host {} are not consecutive: a host's main process serves one run \
                 of its group's ranks",
                rank.host
            );
            order.push(rank.host.clone());
            previous = Some(rank.host.as_str());
        }
    }
    anyhow::ensure!(
        order.first() == Some(engine_host),
        "the deployment's first rank runs on {:?} and its engine host is {engine_host}: the engine \
         host's main process holds rank 0 and the schedulers",
        order.first()
    );
    let mut first = 0_u32;
    let mut hosts = Vec::with_capacity(order.len());
    for host in order {
        let devices = local_devices(&host, config)?;
        let count = u32::try_from(devices.len()).context("a host of more than u32 ranks")?;
        hosts.push(HostRanks {
            host,
            first,
            devices,
        });
        first += count;
    }
    Ok(hosts)
}

/// The launch file's contents for the process `engine` on `host`, in a world of `world_size`
/// ranks: `args`, with the process's place and the interfaces `interfaces` states. The node that
/// starts it states its own Build Server (`Spawn::launch`).
fn place_on(
    mut args: EngineArgs,
    engine: u64,
    interfaces: &Interfaces<'_>,
    host: &HostRanks,
    world_size: u32,
) -> anyhow::Result<EngineArgs> {
    let config = &args.engine_config;
    anyhow::ensure!(
        !config.model.model.is_empty(),
        "the engine configuration names no model"
    );
    let queue_depth = u32::try_from(config.device.queue_depth.get())
        .context("`device_config.queue_depth` exceeds the launch file's field")?;
    let scheduler_node = host.host == config.parallel.engine_host;
    // A new epoch takes the identity of its first engine; an engine that
    // replaces a lost one goes on with its epoch.
    if args.epoch == 0 {
        args.epoch = engine;
    }
    args.world_size = world_size;
    args.host = host.host.clone();
    args.requests = interfaces.requests.to_owned();
    args.processes = interfaces.processes.to_owned();
    args.engine = engine;
    args.build_server = interfaces.build_server.display().to_string();
    args.devices = host.devices.clone();
    args.group = ENGINE_GROUP.to_owned();
    args.rank = host.first;
    args.scheduler_node = scheduler_node;
    // The world is stated by the start that reserves its sockets.
    args.world = None;
    // The bound is the epoch's, and the epoch states it in its engine
    // configuration, which the engine reads.
    args.watchdog_timeout_s = None;
    args.queue_depth = queue_depth;
    Ok(args)
}

/// The CUDA ordinals `host`'s ranks serve, in rank order and without repeats.
///
/// A deployment that states no `device_config.devices` still gives each
/// host's ranks the ordinals the shorthand numbers them from zero; one that
/// does state an order has already had it folded into the ranks by
/// [`proto::config::WorkerGroupConfig::shorthand`]. Reading the ranks is the
/// one account of both, and it is what the launch file states as the devices
/// of this node's process (§2.3, §7.4).
///
/// # Errors
///
/// Fails when a rank of `host` does not run on a CUDA device, or when no rank
/// of the deployment runs on `host`.
fn local_devices(host: &str, config: &proto::config::EngineConfig) -> anyhow::Result<Vec<u32>> {
    let mut devices = Vec::new();
    for group in &config.parallel.worker_groups {
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
        "the rank layout names no device for this rank: no rank of the deployment runs on {host}"
    );
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::ServerArgs;

    /// A launch places the scheduler node's process on the engine host's devices, as the
    /// deployment's ranks name them, with the interfaces the leader states and none it does not.
    #[test]
    fn a_launch_places_the_process_on_the_engine_hosts_devices() {
        let mut options = ServerArgs {
            model: Some("/models/qwen3".to_owned()),
            host_identity: "rank-0".to_owned(),
            tensor_parallel_size: std::num::NonZeroUsize::new(2).unwrap(),
            ..Default::default()
        };
        options.device_config.devices = vec![6, 7];
        let args = EngineArgs::new(
            options.engine_config().expect("the configuration resolves"),
            crate::epoch::modules_of(&options),
        );
        let interfaces = Interfaces {
            requests: "",
            processes: "",
            build_server: Path::new(""),
        };

        let placed = place(args, 1, &interfaces).expect("the process places");
        assert_eq!(placed.devices, vec![6, 7]);
        assert_eq!(placed.host, "rank-0");
        assert_eq!(placed.engine, 1);
        // A new epoch takes its first engine's number.
        assert_eq!(placed.epoch, 1);
        assert_eq!((placed.rank, placed.world_size), (0, 2));
        assert!(placed.scheduler_node);
        assert!(placed.world.is_none());
        assert!(placed.requests.is_empty() && placed.processes.is_empty());
        assert!(placed.build_server.is_empty());
    }

    /// A worker group whose ranks spread over two hosts is one main process per host, the engine
    /// host's first and holding rank 0, each with its own run of the group's ranks.
    #[test]
    fn a_group_over_two_hosts_is_one_process_per_host() {
        let mut options = ServerArgs {
            model: Some("/models/qwen3".to_owned()),
            host_identity: "rank-0".to_owned(),
            tensor_parallel_size: std::num::NonZeroUsize::new(4).unwrap(),
            ..Default::default()
        };
        options.device_config.worker_hosts = vec!["rank-1".to_owned()];
        let config = options.engine_config().expect("the configuration resolves");
        let hosts = world_hosts(&config).expect("the hosts of the group");
        assert_eq!(
            hosts,
            vec![
                HostRanks {
                    host: "rank-0".to_owned(),
                    first: 0,
                    devices: vec![0, 1],
                },
                HostRanks {
                    host: "rank-1".to_owned(),
                    first: 2,
                    devices: vec![0, 1],
                },
            ]
        );
    }

    /// The epoch's module set every launch file states.
    fn modules() -> proto::module::ModuleSet {
        crate::epoch::modules_of(&ServerArgs::default())
    }

    /// An epoch update's target starts from a launch that states its own launch generation, the
    /// epoch, the daemon's interfaces and the epoch's module set; one that names the engine it
    /// succeeds starts parked, and one that names none serves.
    #[test]
    fn a_targets_launch_states_its_generation_its_modules_and_whether_it_parks() {
        let directory = tempfile::tempdir().unwrap();
        let control = EngineControl::new(
            &directory.path().join("requests.sock"),
            "unix:/run/processes.sock".into(),
            "rank-0".into(),
            "model".into(),
            std::path::PathBuf::from("/run/build-server.sock"),
        )
        .unwrap();
        let mut options = ServerArgs {
            model: Some("/models/qwen3".to_owned()),
            served_model_name: Some("model".to_owned()),
            host_identity: "rank-0".to_owned(),
            ..Default::default()
        };
        options.device_config.devices = vec![0];
        options.device_config.queue_depth = std::num::NonZeroUsize::new(4).unwrap();
        let mut args = EngineArgs::new(
            options.engine_config().expect("the configuration resolves"),
            modules(),
        );
        args.epoch = 3;

        let (launch, _) = control.target_launch(&args, 9).unwrap();
        assert_eq!(launch.engine, 9);
        assert_eq!(launch.epoch, 3);
        assert_eq!(launch.processes, "unix:/run/processes.sock");
        assert_eq!(launch.requests, control.request_address());
        assert_eq!(launch.queue_depth, 4);
        assert_eq!(launch.world_size, 1);
        assert_eq!(launch.modules, modules());
        assert!(!launch.parked());

        args.predecessor = Some(wire::Predecessor {
            epoch: 3,
            segments: Vec::new(),
        });
        let (parked, _) = control.target_launch(&args, 10).unwrap();
        assert!(parked.parked());
        assert_eq!(parked.epoch, 3);
    }

    /// A handed-over control connection whose accepted side is gone closed with the
    /// predecessor's service, and that is the predecessor leaving, not an engine exit:
    /// the successor waits for the engine's own dial instead of refusing the adoption
    /// (§7.5.1 step 2).
    #[test]
    fn a_closed_control_connection_is_not_an_engine_exit() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("processes.sock");
        let address = format!("unix:{}", path.display());
        let listener = proto::Listener::bind(&address).unwrap();
        // The predecessor's client end, whose accepted side closed with its service.
        let control = proto::Connection::connect(&address).unwrap();
        drop(listener.try_accept().expect("the accepted connection"));
        drop(listener);

        let mut control = control;
        assert!(
            !still_serving(&mut control).expect("a closed connection is not an engine exit"),
            "a closed control connection reports that the daemon's own client is gone"
        );
    }
}
