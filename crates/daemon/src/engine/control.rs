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
use proto::process::{self, command, downstream, event, hello, reply, report, request, upstream};
use proto::request::wire;
use proto::{Connection, Listener};

use crate::engine::client::{EngineClient, EngineInfo, EngineState, StatsSink};

/// The worker group a daemon supervises an engine process under.
pub(crate) const ENGINE_GROUP: &str = "engine";

/// How long a switch's target may take to load and state what it is.
const TARGET_START: Duration = Duration::from_secs(600);
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
        let standby = launches[0].standby();

        // Every process starts before any is awaited: the scheduler node's waits for its peers.
        let mut controls = Vec::with_capacity(launches.len());
        for (launch, inherit) in launches.iter().zip(inherit) {
            controls.push(self.spawn(launch, inherit)?);
        }
        let mut links = Vec::with_capacity(launches.len());
        for (launch, control) in launches.iter().zip(controls.iter_mut()) {
            links.push(self.welcome(launch.engine, control)?);
        }
        let started = await_started(&mut links, &mut controls, &launches, standby)?;

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

    pub fn request_address(&self) -> &str {
        &self.request_address
    }

    /// The identity the next engine this control starts takes.
    pub fn next_engine(&self) -> u64 {
        self.next_engine.load(Ordering::Relaxed)
    }

    /// Joins an epoch switch's target, which executes `epoch`: the child the switch started dials
    /// this daemon's request socket, is welcomed, and states what it is. The switch commands it on
    /// the process link, so this connection carries only its requests from `Serve` on.
    ///
    /// # Errors
    ///
    /// Fails when the child does not connect in time, speaks another major version, or fails to
    /// start.
    pub fn join_target(
        &self,
        engine: u64,
        epoch: u64,
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
        let (mut reader, mut writer, hello) = self.accept(engine, None)?;
        let refusal = if hello.major != proto::PROTOCOL_MAJOR {
            format!(
                "the engine speaks major version {} and this daemon {}",
                hello.major,
                proto::PROTOCOL_MAJOR
            )
        } else {
            String::new()
        };
        writer.send(&wire::Welcome {
            major: proto::PROTOCOL_MAJOR,
            build: proto::BUILD.to_owned(),
            refusal: refusal.clone(),
        })?;
        anyhow::ensure!(refusal.is_empty(), "refused the engine: {refusal}");
        let deadline = Instant::now() + TARGET_START;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            anyhow::ensure!(
                !remaining.is_zero(),
                "engine {engine} stated nothing in time"
            );
            let Some(message) = reader.recv::<wire::FromEngine>(remaining)? else {
                continue;
            };
            let Some(wire::from_engine::Kind::State(state)) = message.kind else {
                continue;
            };
            let known = match wire::EngineState::try_from(state.state) {
                Ok(wire::EngineState::Standby) => EngineState::Standby,
                Ok(wire::EngineState::Ready) => EngineState::Ready,
                Ok(wire::EngineState::Failed) => {
                    anyhow::bail!("the target failed to start: {}", state.error)
                }
                _ => continue,
            };
            let info = state.info.context("the engine reported no info")?;
            tracing::info!(engine, build = %hello.build, "the switch's target joined this daemon");
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
            major: proto::PROTOCOL_MAJOR,
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

    /// What the epoch layer starts one epoch's main process with: the
    /// executable the target names, at `generation`, and the arguments every
    /// engine process of this daemon gets.
    ///
    /// The build is named only when `executable` is this daemon's own binary.
    /// Another executable's children vouch for themselves: the first one's
    /// `Hello` fixes the identity every other child of the epoch must report.
    pub(crate) fn epoch_start(
        &self,
        args: &EngineArgs,
        engine: u64,
    ) -> anyhow::Result<crate::updates::switch::Start> {
        anyhow::ensure!(
            world_hosts(&args.engine_config)?.len() == 1,
            "an epoch switch starts the target's main processes from one launch, and a worker \
             group with ranks on several hosts starts one per host: such a deployment changes \
             epochs by a relaunch"
        );
        let launch = self.launch(args.clone(), engine)?;
        let executable = launch
            .engine_config
            .worker
            .executable
            .to_string_lossy()
            .into_owned();
        Ok(crate::updates::switch::Start {
            build: own_executable(&executable).then(|| proto::BUILD.to_owned()),
            executable,
            launch,
            environment: main_process_environment(),
        })
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

        let executable = launch
            .engine_config
            .worker
            .executable
            .to_string_lossy()
            .into_owned();
        control.send(&process::Upstream {
            kind: Some(upstream::Kind::Request(process::Request {
                id: 1,
                kind: Some(request::Kind::Spawn(process::Spawn {
                    host: host.to_owned(),
                    group: ENGINE_GROUP.to_owned(),
                    rank: launch.rank,
                    generation: engine,
                    executable,
                    arguments: vec!["engine".to_owned()],
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
    standby: bool,
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
                    Ok(reported @ (wire::EngineState::Ready | wire::EngineState::Standby))
                        if (reported == wire::EngineState::Standby) == standby =>
                    {
                        let info = state.info.context("the engine reported no info")?;
                        let state = if standby {
                            tracing::info!(
                                engine,
                                host,
                                "the engine is prepared beside its predecessor"
                            );
                            EngineState::Standby
                        } else {
                            tracing::info!(
                                engine,
                                host,
                                schedulers = info.schedulers,
                                "the engine serves"
                            );
                            EngineState::Ready
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
    let pinned = prepare_at(leader_server, Selector::Latest, filled, descriptors, steps)?;
    for server in servers {
        if server.host == leader {
            continue;
        }
        let selected = prepare_at(server, Selector::Fixed, filled, descriptors, steps)?;
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

/// `filled` with the packages the leader's store holds for every operator the deployment names
/// nothing for: the generation's full module set.
///
/// A deployment names the operators its configuration selects a backend for; for every other
/// operator the Build Server selects, for each descriptor key, among its packages (`cover`). So
/// what such an operator runs is decided by the packages the store holds, and the epoch's record
/// keeps them - each backend's latest installed revision, one slot per backend with that backend
/// as its part - so a revision installed for one of them is a package change a configuration
/// comparison sees. These slots are not named to a main process (`GenerationSwitch::precopy`): they
/// are what the selection chooses among, not a choice.
///
/// # Errors
///
/// Fails when the leader's Build Server cannot be reached or answers no listing.
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
    let listing = client
        .list()
        .with_context(|| format!("reading the store of {}", leader_server.host))?;
    let mut complete = filled.clone();
    for package in listing.packages {
        if package.op == POLICY_SLOT
            || proto::module::operator::named(&package.op).is_none()
            || filled.kernels.keys().any(|slot| slot.op == package.op)
        {
            continue;
        }
        // Each backend's latest revision: `prepare` selects the same one for the slot.
        complete.kernels.insert(
            proto::module::ModuleSlot {
                op: package.op,
                phase: None,
                part: Some(package.backend.clone()),
            },
            proto::module::PackageRevision {
                backend: package.backend,
                revision: 0,
                arguments: toml::Table::new(),
            },
        );
    }
    Ok(complete)
}

/// How one host's `prepare` names the package that fills an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Selector {
    /// The backend's latest installed revision.
    Latest,
    /// Exactly this package, revision and all.
    Fixed,
}

/// Runs `prepare` at one host's Build Server, selecting each slot the way
/// `selector` says, and answers the epoch's modules with the revisions and
/// parameters it selected.
fn prepare_at(
    server: &BuildServer,
    selector: Selector,
    filled: &crate::epoch::ModuleSet,
    descriptors: &[build_server::Descriptors],
    steps: &PrepareSteps,
) -> anyhow::Result<crate::epoch::ModuleSet> {
    let policy_slot = proto::module::ModuleSlot::all(POLICY_SLOT);
    let mut requests = Vec::with_capacity(filled.kernels.len() + 1);
    for (slot, package) in &filled.kernels {
        requests.push(build_server::PackageRequest {
            slot: slot.clone(),
            selector: selected(selector, package),
        });
    }
    requests.push(build_server::PackageRequest {
        slot: policy_slot.clone(),
        selector: selected(selector, &filled.policy),
    });
    let request = build_server::PrepareRequest {
        targets: server.targets.clone(),
        packages: requests,
        descriptors: descriptors.to_vec(),
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
        prepared = prepared.packages.len(),
        "a Build Server answered"
    );
    let mut kernels = BTreeMap::new();
    let mut policy = None;
    for chosen in prepared.packages {
        if chosen.slot == policy_slot {
            policy = Some(chosen.package);
        } else {
            kernels.insert(chosen.slot, chosen.package);
        }
    }
    let policy = policy.context("the Build Server selected no scheduling policy package")?;
    anyhow::ensure!(
        kernels.len() == filled.kernels.len(),
        "the Build Server of {} selected modules for {} of the {} slots asked",
        server.host,
        kernels.len(),
        filled.kernels.len()
    );
    Ok(crate::epoch::ModuleSet { kernels, policy })
}

/// The slot the scheduling policy package fills: the name the policy packages state
/// as their `op` (`crates/scheduling/policies/<backend>/package.toml`). The
/// scheduling interface is not an operator (§5.1): the epoch record keeps its
/// package beside the kernels' rather than among them, so `prepare` names it
/// by this slot and the daemon takes it back out.
const POLICY_SLOT: &str = "scheduling";

/// The selector a `prepare` request uses for `package`.
fn selected(
    selector: Selector,
    package: &proto::module::PackageRevision,
) -> build_server::Selector {
    match selector {
        Selector::Latest => build_server::Selector::Latest {
            backend: package.backend.clone(),
            arguments: package.arguments.clone(),
        },
        Selector::Fixed => build_server::Selector::Fixed(package.clone()),
    }
}

/// One epoch's main process on a node, and the daemon's process connection
/// to it: the connection its `Report`s arrive on and the `Command`s go down
/// (§2.3, §7.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpochProcess {
    /// The node the main process runs on.
    pub host: String,
    /// The engine number its process connection is registered under, which a
    /// command names and a report carries.
    pub engine: u64,
}

/// How long a main process takes to prepare a generation's modules: it
/// resolves and loads the modules, instantiates the policy, rebinds or
/// recaptures the graphs and shadows the replaced operators (§7.3 step 2).
const PRECOPY_TIMEOUT: Duration = Duration::from_secs(20 * 60);

impl crate::interference::Throttle for EngineControl {
    fn throttle(&self, halve: bool) -> Result<Vec<String>, String> {
        self.throttle_transfers(halve)
            .map_err(|error| format!("{error:#}"))
    }
}

/// How long a main process takes to switch, commit, revert or purge: each is
/// a boundary decision the main process makes at its next batch (§7.3 steps
/// 3-6). A switchover also warms the generation, one capture per bucket of
/// the deployment's catalogue, and its wait is measured from the last
/// `Warming` report rather than from the command: the bound is on a capture
/// that stops moving, never on how many captures a deployment has.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

/// The generation layer of a change (§7.3): it moves the serving main processes
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
pub struct GenerationSwitch<'a> {
    /// The daemon's command channel to the main processes (§2.3).
    commands: &'a crate::supervision::ProcessCommands,
    processes: Vec<EpochProcess>,
    update: crate::epoch::Update,
    /// The module generation each main process numbered for the precopy it
    /// answered, in the processes' order: the main process numbers its
    /// generations, the daemon sends them back in `Switchover` and `Purge`.
    /// Empty before a precopy.
    generations: Vec<u64>,
}

impl<'a> GenerationSwitch<'a> {
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

    /// Prepares the next module generation in every main process: the kernel
    /// packages and the scheduling policy package, resolved and loaded, the policy
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
        modules: &BTreeMap<crate::epoch::KernelKey, crate::epoch::PackageRevision>,
        policy: &crate::epoch::PackageRevision,
    ) -> Result<(), crate::epoch::Failure> {
        // A slot with a part is a package the store holds for an operator the deployment names
        // nothing for (`with_store_modules`): the record keeps it, and the main process's
        // selection chooses among such packages itself, so it is not named.
        let command = process::Command {
            kind: Some(command::Kind::Precopy(process::Precopy {
                kernels: modules
                    .iter()
                    .filter(|(slot, _)| slot.part.is_none())
                    .map(|(slot, package)| crate::epoch::kernel_to_wire(slot, package))
                    .collect(),
                policy: Some(crate::epoch::package_to_wire(policy)),
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
        }
        let prepared = await_reports(
            commands,
            processes,
            update,
            PRECOPY_TIMEOUT,
            "Precopy",
            |process, report| match report.kind.as_ref() {
                Some(report::Kind::Precopied(precopied)) => {
                    // The answer carries the operators this update could
                    // not compare, and an empty list is the statement that
                    // every replaced operator was (process.proto,
                    // `Precopied`). Nothing else prints it: a generation
                    // with no shadow evidence for part of what it changed
                    // would read exactly like one with evidence for all of
                    // it, which is what an update's trial rests on.
                    if !precopied.uncompared.is_empty() {
                        tracing::warn!(
                            host = %process.host,
                            engine = process.engine,
                            uncompared = ?precopied.uncompared,
                            "a generation was prepared with operators it could not compare",
                        );
                    }
                    Some(Ok(precopied.generation))
                }
                Some(report::Kind::PrecopyFailed(failed)) => Some(Err(crate::epoch::Failure {
                    kind: "precopy_failed".to_owned(),
                    reason: failed.reason.clone(),
                    device: None,
                })),
                _ => None,
            },
        );
        match prepared {
            Ok(prepared) => *generations = prepared,
            Err(failure) => {
                // What was prepared is not switchable as a set: the
                // serving generation stays, and no generation moves.
                generations.clear();
                return Err(failure);
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
        // Every process switches before any answer is awaited: a switchover warms the generation
        // up, and the ranks of a group that spans processes capture their collectives together.
        for (process, generation) in self.processes.iter().zip(self.generations.clone()) {
            send(
                self.commands,
                process,
                process::Command {
                    kind: Some(command::Kind::Switchover(process::Switchover {
                        generation,
                    })),
                },
                "Switchover",
            )?;
        }
        let generations: BTreeMap<u64, u64> = self
            .processes
            .iter()
            .map(|process| process.engine)
            .zip(self.generations.iter().copied())
            .collect();
        await_reports(
            self.commands,
            &self.processes,
            &mut self.update,
            COMMAND_TIMEOUT,
            "Switchover",
            |process, report| match report.kind.as_ref() {
                Some(report::Kind::Switched(switched)) => Some(answered(
                    report,
                    generations.get(&process.engine) == Some(&switched.generation),
                )),
                // A generation that cannot warm up is answered with a
                // refusal, never with `Switched`: discarded before it took
                // over (`PrecopyFailed`) or reverted after it did
                // (`Reverted`). Waiting for the switchover answer would
                // turn a named engine failure into two minutes of silence
                // and then a timeout.
                Some(report::Kind::PrecopyFailed(failed)) => Some(Err(crate::epoch::Failure {
                    kind: "precopy_failed".to_owned(),
                    reason: failed.reason.clone(),
                    device: None,
                })),
                Some(report::Kind::Reverted(reverted)) => Some(Err(crate::epoch::Failure {
                    kind: "reverted".to_owned(),
                    reason: reverted.reason.clone(),
                    device: None,
                })),
                _ => None,
            },
        )?;
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
        for process in self.processes.iter() {
            send(self.commands, process, command.clone(), "Commit")?;
        }
        await_reports(
            self.commands,
            &self.processes,
            &mut self.update,
            COMMAND_TIMEOUT,
            "Commit",
            |_, report| {
                matches!(report.kind.as_ref(), Some(report::Kind::Committed(_))).then_some(Ok(()))
            },
        )?;
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
        let mut sent = Vec::with_capacity(self.processes.len());
        for process in self.processes.iter() {
            match send(self.commands, process, command.clone(), "Revert") {
                Ok(()) => sent.push(process.clone()),
                Err(failure) => {
                    self.update.fail(failure.clone());
                    outcome = Err(failure);
                }
            }
        }
        let answered = await_reports(
            self.commands,
            &sent,
            &mut self.update,
            COMMAND_TIMEOUT,
            "Revert",
            |_, report| {
                matches!(report.kind.as_ref(), Some(report::Kind::Reverted(_))).then_some(Ok(()))
            },
        );
        if let Err(failure) = answered {
            self.update.fail(failure.clone());
            outcome = Err(failure);
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
                process,
                process::Command {
                    kind: Some(command::Kind::Purge(process::Purge { generation })),
                },
                "Purge",
            )?;
        }
        await_reports(
            self.commands,
            &self.processes,
            &mut self.update,
            COMMAND_TIMEOUT,
            "Purge",
            |_, report| {
                matches!(report.kind.as_ref(), Some(report::Kind::Purged(_))).then_some(Ok(()))
            },
        )?;
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

/// Reads the reports the main processes send until `matches` took one from each of `processes`:
/// the answers to the command every one of them was sent, in `processes`' order, or the failure
/// that ends the wait. A failure report from any main process fails the update with the first such
/// failure (§7.3 step 4); a report of a process this wait does not list, or of one that answered
/// already, is not this wait's. A `Warming` report of a listed process is its warmup moving on to
/// another capture, and restarts `timeout`: the wait ends on a main process that stops making
/// progress, not on a long catalogue.
///
/// The command goes to every process before any answer is awaited, so the processes of a worker
/// group that spans hosts carry it out together: a switchover's warmup captures collectives every
/// rank must reach.
fn await_reports<T>(
    commands: &crate::supervision::ProcessCommands,
    processes: &[EpochProcess],
    update: &mut crate::epoch::Update,
    timeout: Duration,
    what: &str,
    matches: impl Fn(&EpochProcess, &process::Report) -> Option<Result<T, crate::epoch::Failure>>,
) -> Result<Vec<T>, crate::epoch::Failure> {
    let _awaiting = commands.awaiting();
    let mut answers: Vec<Option<T>> = processes.iter().map(|_| None).collect();
    let mut deadline = Instant::now() + timeout;
    while let Some(waiting) = answers.iter().position(Option::is_none) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(unanswered(&processes[waiting], timeout, what));
        }
        let Some((engine, report)) = commands.report(remaining) else {
            return Err(unanswered(&processes[waiting], timeout, what));
        };
        if let Some(failure) = crate::epoch::Failure::of_report(&report) {
            update.fail(failure.clone());
            return Err(failure);
        }
        let Some(index) = processes
            .iter()
            .position(|process| process.engine == engine)
            .filter(|&index| answers[index].is_none())
        else {
            tracing::debug!(
                engine,
                command = what,
                "a main process reported while a switch waited"
            );
            continue;
        };
        let process = &processes[index];
        if let Some(process::report::Kind::Warming(warming)) = report.kind.as_ref() {
            tracing::info!(
                host = %process.host,
                engine = process.engine,
                generation = warming.generation,
                step = %warming.step,
                command = what,
                "a main process's warmup moved on"
            );
            deadline = Instant::now() + timeout;
            continue;
        }
        if let Some(outcome) = matches(process, &report) {
            match outcome {
                Ok(answer) => {
                    tracing::info!(
                        host = %process.host,
                        engine = process.engine,
                        command = what,
                        "a main process answered"
                    );
                    answers[index] = Some(answer);
                }
                Err(failure) => {
                    update.fail(failure.clone());
                    return Err(failure);
                }
            }
        }
    }
    Ok(answers.into_iter().flatten().collect())
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
    // Both are the epoch's, and the epoch states them in its engine
    // configuration, which the child reads.
    args.watchdog_timeout_s = None;
    args.rollback_s = None;
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
    use crate::engine::stand_in::{Answer, Seen, main_processes, process};
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
        let args = EngineArgs::new(options.engine_config().expect("the configuration resolves"));
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
            std::path::PathBuf::from("/run/build-server.sock"),
        )
        .unwrap();
        let own = std::env::current_exe().unwrap();
        // The deployment's own engine configuration, as the options resolve it, on this control's
        // host: an epoch's start reads the ranks it places.
        let mut options = ServerArgs {
            model: Some("/models/qwen3".to_owned()),
            served_model_name: Some("model".to_owned()),
            host_identity: "rank-0".to_owned(),
            ..Default::default()
        };
        options.device_config.devices = vec![0];
        options.device_config.queue_depth = std::num::NonZeroUsize::new(4).unwrap();
        options.worker_config.executable = own;
        let mut args =
            EngineArgs::new(options.engine_config().expect("the configuration resolves"));
        args.epoch = 3;

        let start = control.epoch_start(&args, 9).unwrap();
        assert_eq!(start.build, Some(proto::BUILD.to_owned()));
        // The file is the adapter's to write for the process it starts; the
        // start carries its contents, with this process's generation.
        assert_eq!(start.launch.engine, 9);
        assert_eq!(start.launch.epoch, 3);
        assert_eq!(start.launch.processes, "unix:/run/processes.sock");
        assert_eq!(start.launch.requests, control.request_address());
        assert_eq!(start.launch.queue_depth, 4);
        assert_eq!(start.launch.world_size, 1);

        // The launch states whether the child stands by (§2.5): a standby has
        // to know the role before it claims storage, and a serving child must
        // not read the file as one.
        assert!(!start.launch.standby());
        args.predecessor = Some(wire::Predecessor {
            epoch: 3,
            segments: Vec::new(),
        });
        let standby = control.epoch_start(&args, 10).unwrap();
        assert!(standby.launch.standby());
        assert_eq!(standby.launch.epoch, 3);
        args.predecessor = None;

        args.engine_config.worker.executable = std::path::PathBuf::from("/opt/uniserve-b");
        let other = control.epoch_start(&args, 9).unwrap();
        assert_eq!(other.build, None);
        assert_eq!(other.executable, "/opt/uniserve-b");
    }

    /// An update of the default deployment's filled modules, for the switch to
    /// move.
    fn update() -> crate::epoch::Update {
        crate::epoch::Update::new(
            1,
            crate::epoch::Epoch::of(ServerArgs::default()).expect("the configuration resolves"),
        )
    }

    /// The generation layer moves every main process through the generations each
    /// one numbered for itself (§7.3): the precopy, then the switchover to the
    /// generation that process answered with, then the commit, and the purge of
    /// the generation each one executes.
    #[test]
    fn a_switch_moves_every_main_process_through_the_generations_they_numbered() {
        let (processes, seen) =
            main_processes(vec![(7, Answer::Serves(11)), (8, Answer::Serves(12))]);
        let target =
            crate::epoch::Epoch::of(ServerArgs::default()).expect("the configuration resolves");
        let mut switch = GenerationSwitch::new(
            &processes,
            vec![process("rank-0", 7), process("rank-1", 8)],
            update(),
        );
        switch
            .precopy(&target.modules.kernels, &target.modules.policy)
            .expect("both main processes prepared the modules");
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

    /// A main process that cannot prepare the modules fails the update and
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
        let target =
            crate::epoch::Epoch::of(ServerArgs::default()).expect("the configuration resolves");
        let mut switch = GenerationSwitch::new(
            &processes,
            vec![process("rank-0", 7), process("rank-1", 8)],
            update(),
        );
        let failure = switch
            .precopy(&target.modules.kernels, &target.modules.policy)
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

    /// A main process's fault while the modules are prepared fails the update
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
        let target =
            crate::epoch::Epoch::of(ServerArgs::default()).expect("the configuration resolves");
        let mut switch = GenerationSwitch::new(&processes, vec![process("rank-0", 7)], update());
        let failure = switch
            .precopy(&target.modules.kernels, &target.modules.policy)
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
    fn a_policy_revert_while_the_modules_are_prepared_fails_the_update() {
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
        let target =
            crate::epoch::Epoch::of(ServerArgs::default()).expect("the configuration resolves");
        let mut switch = GenerationSwitch::new(&processes, vec![process("rank-0", 7)], update());
        let failure = switch
            .precopy(&target.modules.kernels, &target.modules.policy)
            .expect_err("a policy revert fails the update");
        assert_eq!(failure.kind, "policy_reverted");
        assert!(failure.reason.contains("priority"), "{}", failure.reason);
    }
}
