//! Starting the deployment's engine process and connecting to it.
//!
//! The leader daemon starts each engine process as a child of a node's
//! supervision, through the process interface like any process of the
//! deployment: [`EngineControl`] connects to the daemon's process socket and
//! asks it to spawn `uniserve engine` on the node, which connects back to the
//! daemon's request socket and says hello. The daemon responds with the
//! engine's specification and waits until the engine reports that it serves;
//! the engine's process lives for as long as its control connection does.
//!
//! A deployment whose worker groups have ranks on several hosts has one engine
//! per host ([`world_hosts`]): the scheduler node's, on the deployment's engine
//! host, and a peer on each other host. They are started together, the daemon
//! having reserved on the scheduler node the two sockets the peers dial (the
//! peer channel and the ranks' host-collective rendezvous), and the engine is
//! the scheduler node's client, which holds its peers' and ends with them.

use proto::config::EngineArgs;
use std::collections::BTreeMap;
use std::os::fd::AsRawFd as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Context as _;
use proto::admin::NodeUsage;
use proto::process::{self, downstream, event, hello, reply, request, upstream};
use proto::request::wire;
use service::{Connection, Listener};

use crate::engine::client::{EngineClient, EngineInfo, EngineState, StatsSink};

/// The worker group a daemon supervises an engine process under.
pub(crate) const ENGINE_GROUP: &str = "engine";

/// How long a Build Server that is replaced may take to stop once it was asked to: it ends its
/// builds' compiler processes, which end at once.
const BUILD_SERVER_STOP: std::time::Duration = std::time::Duration::from_secs(30);

/// Makes sure the Build Server of `deployment` serves on this node with `settings`, and returns
/// its socket in the node's runtime directory `runtime` ([`build_server::socket`]): when none
/// serves, the deployment's daemon starts it with
/// `settings` from the running build ([`crate::supervision::RUNNING_IMAGE`],
/// `service::host::start`), and it outlives this daemon. A server of the deployment that serves
/// with other settings, which an earlier daemon of the deployment started, is the deployment's
/// own: it is stopped and started again with `settings`. The start ends with the server's
/// readiness record, or with the server's end before it served.
///
/// # Errors
///
/// Fails when the deployment's Build Server cannot be stopped or started.
pub fn ensure_build_server(
    runtime: &service::RuntimeDirectory,
    deployment: &str,
    settings: &build_server::command::Settings,
) -> anyhow::Result<PathBuf> {
    let socket = build_server::socket(runtime, deployment)
        .with_context(|| format!("finding the Build Server socket of {deployment}"))?;
    let mut arguments = vec!["--deployment".to_owned(), deployment.to_owned()];
    arguments.extend(settings.arguments());
    let start = || -> anyhow::Result<build_server::Readiness> {
        ::service::host::start(
            Path::new(crate::supervision::RUNNING_IMAGE),
            build_server::SERVICE,
            &socket,
            &arguments,
            None,
        )
        .with_context(|| format!("starting the Build Server of {deployment}"))
    };
    let record = start()?;
    if record.settings != *settings {
        tracing::info!(
            deployment,
            pid = record.pid,
            "the deployment's Build Server serves with other settings: it is replaced"
        );
        ::service::host::stop(&socket, BUILD_SERVER_STOP)
            .with_context(|| format!("stopping the Build Server of {deployment}"))?;
        start()?;
    }
    Ok(socket)
}

pub struct EngineControl {
    /// The request socket engine processes connect back to.
    requests: Listener,
    request_address: String,
    /// The daemon's process socket.
    processes: service::Endpoint,
    /// This node's host identity, where the engine process runs.
    host: String,
    /// The deployment the engines serve.
    deployment: String,
    /// This node's Build Server socket, which every engine process is told
    /// to resolve an epoch's modules from.
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
        let processes: service::Endpoint = processes.parse()?;
        // The request socket is reached as the process socket is: a Unix socket on one host (in
        // this daemon's own instance directory, so no other process's socket lies at its path),
        // the same interface's TCP endpoint, on a port the kernel picks, across hosts. Its
        // listener removes a Unix socket's file when it closes.
        let endpoint = match &processes {
            service::Endpoint::Unix(_) => service::Endpoint::unix(socket),
            service::Endpoint::Tcp { host, .. } => service::Endpoint::Tcp {
                host: host.clone(),
                port: 0,
            },
        };
        let requests = Listener::bind(&endpoint)
            .with_context(|| format!("binding the request socket {endpoint}"))?;
        let request_address = requests.endpoint().to_string();
        Ok(Self {
            requests,
            request_address,
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
    /// the engine package its launch's module set names, which the node that starts it resolves at
    /// its own Build Server. This control states the process's place and interfaces in the launch
    /// ([`place_world`]). Blocks for the engine's whole start, its workers' loading included.
    ///
    /// # Errors
    ///
    /// Fails when the daemon cannot start the process, the process exits or
    /// does not connect back, its build speaks a major version this daemon
    /// does not, or it reports that its start failed.
    pub fn start(&self, args: EngineArgs, stats: StatsSink) -> anyhow::Result<EngineClient> {
        let engine = self.launch_number();
        self.start_at(args, engine, stats)
    }

    /// The next launch number of the deployment: the identity of one engine process, which its
    /// launch states, its process link's hello states and every command and report names.
    ///
    /// This control is the deployment's one allocator of them: every engine it starts, a
    /// standby, the engines of an epoch update's target and an adopted deployment's later
    /// engines all take their number here, so one number names one process across the
    /// deployment's hosts.
    pub(crate) fn launch_number(&self) -> u64 {
        self.next_engine.fetch_add(1, Ordering::Relaxed)
    }

    /// [`EngineControl::start`] under a number the caller took from [`Self::launch_number`]
    /// (a standby's, which the standby rules record before it starts).
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
        let hosts = world_hosts(&args.engine_config)?;
        // The peers' numbers follow the scheduler node's from the same counter, so one number names
        // one process of the deployment.
        let numbers: Vec<u64> = std::iter::once(engine)
            .chain((1..hosts.len()).map(|_| self.launch_number()))
            .collect();
        self.start_world(args, &numbers, stats)
    }

    /// Starts the deployment of `args` as one engine per host of its worker group, numbered
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
        let (launches, inherit) = self.world(args, numbers)?;
        let epoch = launches[0].epoch;
        let parked = launches[0].parked();

        // Every process starts before any is awaited: the scheduler node's waits for its peers.
        let mut controls = Vec::with_capacity(launches.len());
        for (launch, inherit) in launches.iter().zip(inherit) {
            controls.push(self.spawn(launch, inherit)?);
        }
        let mut links = Vec::with_capacity(launches.len());
        let mut versions = Vec::with_capacity(launches.len());
        for (launch, control) in launches.iter().zip(controls.iter_mut()) {
            let (reader, writer, version) = self.welcome(launch.engine, control)?;
            links.push((reader, writer));
            versions.push(version);
        }
        let started = await_started(&mut links, &mut controls, &launches, parked)?;

        let mut clients = Vec::with_capacity(launches.len());
        let mut sinks = std::iter::once(stats);
        for ((((reader, writer), control), version), (launch, (info, state))) in links
            .into_iter()
            .zip(controls)
            .zip(versions)
            .zip(launches.iter().zip(started))
        {
            // A peer holds no scheduler, so it reports no statistics.
            let sink = sinks.next().unwrap_or_else(|| Box::new(|_| {}));
            clients.push(
                EngineClient::new(reader, writer, epoch, info, state, sink, Some(control))?
                    .numbered(launch.engine)
                    .on_host(&launch.host)
                    .speaking(version),
            );
        }
        let mut clients = clients.into_iter();
        let scheduler = clients.next().context("a deployment of no engine")?;
        Ok(scheduler.with_peers(clients.collect()))
    }

    /// The launch files of the world of `args` ([`place_world`]), numbered `numbers` (the
    /// scheduler node's first), and the reservations each process inherits. A world of several
    /// hosts meets at two sockets reserved on the scheduler node, which that node's process
    /// inherits and every launch names.
    ///
    /// # Errors
    ///
    /// Fails as [`place_world`] fails, and when the scheduler node does not reserve the sockets.
    fn world(
        &self,
        args: EngineArgs,
        numbers: &[u64],
    ) -> anyhow::Result<(Vec<EngineArgs>, Vec<Vec<u64>>)> {
        let processes = self.processes.to_string();
        let interfaces = Interfaces {
            deployment: &self.deployment,
            requests: &self.request_address,
            processes: &processes,
            build_server: &self.build_server,
        };
        let mut launches = place_world(args, numbers, &interfaces)?;

        let mut inherit = vec![Vec::new(); launches.len()];
        if launches.len() > 1 {
            let (channel, store) = self.reserve_world(&launches[0].host)?;
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
                hosts = ?launches.iter().map(|launch| launch.host.as_str()).collect::<Vec<_>>(),
                engines = ?numbers,
                channel = %world.channel,
                store = %world.store,
                "a world whose worker group spans hosts"
            );
        }
        Ok((launches, inherit))
    }

    /// Reserves on `host` the world's two sockets, which the scheduler node's process inherits:
    /// its peer channel's listener and its ranks' host-collective rendezvous.
    fn reserve_world(&self, host: &str) -> anyhow::Result<(process::Reserved, process::Reserved)> {
        let (mut control, _) = self.connect()?;
        let mut reserve = |id| -> anyhow::Result<process::Reserved> {
            control.deliver(&process::Upstream {
                kind: Some(upstream::Kind::Request(process::Request {
                    id,
                    kind: Some(request::Kind::Reserve(process::Reserve {
                        host: host.to_owned(),
                        group: ENGINE_GROUP.to_owned(),
                        engine: None,
                        routable: true,
                    })),
                })),
            })?;
            let reply = await_reply(&mut control, id).context("awaiting a reservation")?;
            anyhow::ensure!(
                reply.error.is_empty(),
                "reserving a socket on {host}: {}",
                reply.error
            );
            match reply.kind {
                Some(reply::Kind::Reserved(reserved)) => Ok(reserved),
                _ => anyhow::bail!("the daemon sent another reply to a reservation"),
            }
        };
        let channel = reserve(1)?;
        let store = reserve(2)?;
        Ok((channel, store))
    }

    /// Accepts engine `engine`'s request link and welcomes it ([`welcome`]); returns the link's
    /// two ends and the version of the engine interface it agreed on.
    fn welcome(
        &self,
        engine: u64,
        control: &mut Connection,
    ) -> anyhow::Result<(Connection, Connection, u32)> {
        let (reader, mut writer, hello) = self.accept(engine, Some(control))?;
        let version = welcome(&mut writer, &hello, engine)?;
        tracing::info!(engine, build = %hello.build, version, "the engine process connected; awaiting its start");
        Ok((reader, writer, version))
    }

    /// The deployment the engines serve, which names its storage at the Storage Servers.
    pub(crate) fn deployment(&self) -> &str {
        &self.deployment
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

    /// The launch number [`Self::launch_number`] hands out next, which a restart in place hands to
    /// the successor ([`Self::adopt`]) and which numbers a deployment's first epoch.
    pub fn next_engine(&self) -> u64 {
        self.next_engine.load(Ordering::Relaxed)
    }

    /// Joins an epoch update's target world, which executes `epoch`: every engine of `processes`
    /// (by host and launch generation, the scheduler node's first) the update started dials this
    /// daemon's request socket, is welcomed, and states that it parked. The update commands them
    /// on their process links, so these connections carry only requests once they serve; the
    /// scheduler node's client holds its peers'. `stats` receives its schedulers' statistics.
    ///
    /// # Errors
    ///
    /// Fails when an engine does not connect in time, speaks no version of the engine interface
    /// this daemon speaks, fails to start, or exits before it stated that it parked.
    pub fn join_target(
        &self,
        processes: &[(String, u64)],
        epoch: u64,
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
        let mut clients = Vec::with_capacity(processes.len());
        let mut sinks = std::iter::once(stats);
        for (host, engine) in processes {
            // A peer holds no scheduler, so it reports no statistics.
            let sink = sinks.next().unwrap_or_else(|| Box::new(|_| {}));
            clients.push(self.join(*engine, epoch, sink)?.on_host(host));
        }
        let mut clients = clients.into_iter();
        let scheduler = clients.next().context("a target of no engine")?;
        Ok(scheduler.with_peers(clients.collect()))
    }

    /// Joins the one target engine `engine` of [`Self::join_target`].
    ///
    /// The start itself has no bound: an engine loads its ranks, resolves its modules and
    /// captures its graphs before it states that it parked, and one that exits meanwhile closes
    /// its request link, which ends the wait ([`await_start`]).
    fn join(&self, engine: u64, epoch: u64, stats: StatsSink) -> anyhow::Result<EngineClient> {
        let (mut reader, mut writer, hello) = self.accept(engine, None)?;
        let version = welcome(&mut writer, &hello, engine)?;
        let (info, known) = await_start(&mut reader)?;
        tracing::info!(engine, build = %hello.build, version, "the update's target joined this daemon");
        EngineClient::new(reader, writer, epoch, info, known, stats, None)
            .map(|client| client.numbered(engine).speaking(version))
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
    /// Fails when the engine's process exits or its request link closes before it states its
    /// state, when it speaks no version of the engine interface this daemon speaks (an engine
    /// attaching to a successor of another build offers the one version it speaks), is another
    /// engine, or reports anything but ready.
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
        let version = welcome(&mut writer, &hello, engine)?;
        loop {
            // An attaching engine states its state right after the welcome.
            let message: wire::FromEngine = reader.recv_next()?;
            if let Some(wire::from_engine::Kind::State(state)) = message.kind {
                anyhow::ensure!(
                    state.state == wire::EngineState::Serving as i32,
                    "engine {engine} attached while it does not serve: {}",
                    state.error
                );
                let info = state.info.context("the engine reported no info")?;
                tracing::info!(engine, build = %hello.build, version, "the engine attached");
                // A daemon adopts the engine of its own node, which runs on this host: the
                // updater names the processes it drains and commands by their host
                // (`EngineClient::processes`), as it does for an engine this control started.
                return EngineClient::new(
                    reader,
                    writer,
                    epoch,
                    info.try_into()?,
                    EngineState::Serving,
                    stats,
                    control,
                )
                .map(|client| {
                    client
                        .numbered(engine)
                        .on_host(&self.host)
                        .speaking(version)
                });
            }
        }
    }

    /// Connects to the daemon's process socket as a client of it, and returns the connection and
    /// the deployment's nodes the daemon's welcome states. The daemon's service answers every
    /// request on it, so each read below ends with the answer or with the connection's end.
    fn connect(&self) -> anyhow::Result<(Connection, Vec<process::Node>)> {
        let mut control = Connection::connect(&self.processes)
            .with_context(|| format!("connecting to the daemon at {}", self.processes))?;
        control.deliver(&process::Hello {
            oldest: proto::ENGINE_INTERFACE.oldest(),
            newest: proto::ENGINE_INTERFACE.newest(),
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
            .recv_next()
            .context("the daemon closed the control connection before its welcome")?;
        if let Some(refusal) = welcome.refusal {
            return Err(
                anyhow::Error::new(refusal).context("the daemon refused the control connection")
            );
        }

        Ok((control, welcome.nodes))
    }

    /// The quotas every node's Storage Server keeps for the deployment, one report per node.
    /// Each node's daemon reads them over its claim, so no engine is asked.
    pub(crate) fn storage_reports(&self) -> anyhow::Result<Vec<NodeUsage>> {
        let (mut connection, nodes) = self.connect()?;
        let mut reports = Vec::with_capacity(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            let host = node.host.clone();
            connection.deliver(&process::Upstream {
                kind: Some(upstream::Kind::Request(process::Request {
                    id: index as u64 + 1,
                    kind: Some(request::Kind::Usage(process::UsageQuery {
                        host: host.clone(),
                        deployment: self.deployment.clone(),
                    })),
                })),
            })?;
            let message: process::Downstream = connection
                .recv_next()
                .context("the daemon closed the connection before it reported its quotas")?;
            let Some(downstream::Kind::Reply(reply)) = message.kind else {
                anyhow::bail!("the daemon did not reply to its quota query");
            };
            anyhow::ensure!(
                reply.error.is_empty(),
                "the quotas of {host}: {}",
                reply.error
            );
            let Some(reply::Kind::Usage(report)) = reply.kind else {
                anyhow::bail!("the daemon returned no quota");
            };
            let usage: storage::protocol::Usage = storage::protocol::decode(&report.usage)
                .with_context(|| format!("reading the quotas of {host}"))?;
            reports.push(NodeUsage {
                host,
                quotas: usage.quotas,
                host_storage: usage.host,
                tasks: usage.tasks,
            });
        }
        Ok(reports)
    }

    /// Publishes the committed `epochs` at the Storage Server of each node of `hosts`, through the
    /// claim of the deployment that node's daemon holds: `hosts` are the nodes that claimed the
    /// epochs, and only those hold storage of theirs to publish. A node that holds no claim of
    /// the deployment, or none of an epoch, refuses, and the publication fails naming it.
    /// Publication is atomic within each Storage Server and idempotent, so a caller may ask
    /// again.
    ///
    /// # Errors
    ///
    /// Fails when the daemon cannot be reached, and at the first node that refuses or does not
    /// reply in time.
    pub fn publish_epochs(&self, epochs: &[u64], hosts: &[String]) -> anyhow::Result<()> {
        let (mut connection, _) = self.connect()?;
        for host in hosts {
            connection.deliver(&process::Upstream {
                kind: Some(upstream::Kind::Request(process::Request {
                    id: 1,
                    kind: Some(request::Kind::PublishEpochs(process::PublishEpochs {
                        host: host.clone(),
                        deployment: self.deployment.clone(),
                        epochs: epochs.to_vec(),
                    })),
                })),
            })?;
            let message: process::Downstream = connection
                .recv_next()
                .context("the daemon closed the connection before it reported epoch publication")?;
            let Some(downstream::Kind::Reply(reply)) = message.kind else {
                anyhow::bail!("the daemon did not reply to epoch publication");
            };
            anyhow::ensure!(
                reply.error.is_empty(),
                "publishing epochs at {host}: {}",
                reply.error
            );
            anyhow::ensure!(
                matches!(reply.kind, Some(reply::Kind::EpochsPublished(_))),
                "the daemon returned no publication result"
            );
        }
        Ok(())
    }

    /// Has every node's daemon halve, or restore when not `halve`, the byte
    /// budgets of the links the deployment's transfers use, and returns the
    /// hosts that took it. A node that holds no claim of the deployment has
    /// nothing to throttle and says so in its reply.
    ///
    /// # Errors
    ///
    /// Fails when a node's daemon cannot be reached or refuses.
    pub(crate) fn throttle_transfers(&self, halve: bool) -> anyhow::Result<Vec<String>> {
        let (mut connection, nodes) = self.connect()?;
        let mut hosts = Vec::with_capacity(nodes.len());
        for node in nodes {
            connection.deliver(&process::Upstream {
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
            let message: process::Downstream = connection
                .recv_next()
                .context("the daemon closed the connection before it reported the throttle")?;
            let Some(downstream::Kind::Reply(reply)) = message.kind else {
                anyhow::bail!("the daemon did not reply to the throttle");
            };
            anyhow::ensure!(
                reply.error.is_empty(),
                "throttling transfers at {}: {}",
                node.host,
                reply.error
            );
            hosts.push(node.host);
        }
        Ok(hosts)
    }

    /// Has every node's daemon have the deployment's Storage Server there serve with
    /// `settings` (`Request::ConfigureStorage`), and returns once each node's server serves
    /// with them: a running server takes the settings it reads when it acts at once, and is
    /// restarted in place with them otherwise.
    ///
    /// # Errors
    ///
    /// Fails when a node's daemon cannot be reached, and at the first node whose server cannot
    /// take the settings, naming it.
    pub(crate) fn configure_storage(&self, settings: &storage::Settings) -> anyhow::Result<()> {
        let (mut connection, nodes) = self.connect()?;
        for (index, node) in nodes.iter().enumerate() {
            connection.deliver(&process::Upstream {
                kind: Some(upstream::Kind::Request(process::Request {
                    id: index as u64 + 1,
                    kind: Some(request::Kind::ConfigureStorage(process::ConfigureStorage {
                        host: node.host.clone(),
                        deployment: self.deployment.clone(),
                        storage: Some(settings.into()),
                    })),
                })),
            })?;
            let reply = await_reply(&mut connection, index as u64 + 1)
                .context("awaiting the Storage Server's settings")?;
            anyhow::ensure!(
                reply.error.is_empty(),
                "the Storage Server of {}: {}",
                node.host,
                reply.error
            );
            anyhow::ensure!(
                matches!(reply.kind, Some(reply::Kind::StorageConfigured(_))),
                "the daemon of {} answered the settings with no result",
                node.host
            );
        }
        Ok(())
    }

    /// Has every node's daemon have the deployment's Storage Server there run `program` as the
    /// deployment's KV program from now on, or none (`Request::ConfigureProgram`): the leader's
    /// node, this one, steps its tiering hook, and every node saves its state in its own
    /// `program-state/`. Returns how each node's instance started, by host; `None` for a node
    /// that runs none.
    ///
    /// # Errors
    ///
    /// Fails when a node's daemon cannot be reached, and at the first node whose server refuses
    /// the program, naming it; the nodes before it run the program.
    pub(crate) fn program(
        &self,
        program: Option<&crate::programs::KvSetup>,
    ) -> anyhow::Result<Vec<(String, Option<storage::protocol::ProgramStart>)>> {
        let (mut connection, nodes) = self.connect()?;
        let mut started = Vec::with_capacity(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            let program = program
                .map(|program| program.wire(node.host == self.host))
                .transpose()?;
            connection.deliver(&process::Upstream {
                kind: Some(upstream::Kind::Request(process::Request {
                    id: index as u64 + 1,
                    kind: Some(request::Kind::ConfigureProgram(process::ConfigureProgram {
                        host: node.host.clone(),
                        deployment: self.deployment.clone(),
                        program,
                    })),
                })),
            })?;
            let reply = await_reply(&mut connection, index as u64 + 1)
                .context("awaiting the Storage Server's KV program")?;
            anyhow::ensure!(
                reply.error.is_empty(),
                "the Storage Server of {}: {}",
                node.host,
                reply.error
            );
            let Some(reply::Kind::ProgramConfigured(configured)) = reply.kind else {
                anyhow::bail!(
                    "the daemon of {} answered the KV program with no result",
                    node.host
                );
            };
            started.push((node.host.clone(), program_start(&configured.start)?));
        }
        Ok(started)
    }

    /// What the deployment's KV program logged and how its outputs fared on every node since
    /// the last call (`Request::ProgramRecords`), each record with its node's host, node by node
    /// and oldest first within a node; each node's server keeps them no longer.
    ///
    /// # Errors
    ///
    /// Fails when a node's daemon cannot be reached, and at the first node whose records cannot
    /// be read, naming it.
    pub(crate) fn program_records(
        &self,
    ) -> anyhow::Result<Vec<(String, storage::protocol::ProgramRecord)>> {
        let (mut connection, nodes) = self.connect()?;
        let mut records = Vec::new();
        for (index, node) in nodes.iter().enumerate() {
            connection.deliver(&process::Upstream {
                kind: Some(upstream::Kind::Request(process::Request {
                    id: index as u64 + 1,
                    kind: Some(request::Kind::ProgramRecords(
                        process::ProgramRecordsQuery {
                            host: node.host.clone(),
                            deployment: self.deployment.clone(),
                        },
                    )),
                })),
            })?;
            let reply = await_reply(&mut connection, index as u64 + 1)
                .context("awaiting the KV program's records")?;
            anyhow::ensure!(
                reply.error.is_empty(),
                "the KV program's records at {}: {}",
                node.host,
                reply.error
            );
            let Some(reply::Kind::ProgramRecords(read)) = reply.kind else {
                anyhow::bail!(
                    "the daemon of {} answered the KV program's records with none",
                    node.host
                );
            };
            let read: Vec<storage::protocol::ProgramRecord> =
                storage::protocol::decode(&read.records).with_context(|| {
                    format!("reading the KV program's records of {}", node.host)
                })?;
            records.extend(read.into_iter().map(|record| (node.host.clone(), record)));
        }
        Ok(records)
    }

    /// The spans every node of the deployment keeps in its rings that ended at or after `since`
    /// (nanoseconds since the Unix epoch), each node's daemon's and Storage Server's together
    /// with the node's host (`Request::Spans`).
    ///
    /// # Errors
    ///
    /// Fails when a node's daemon cannot be reached or cannot read them.
    pub(crate) fn node_spans(
        &self,
        since: u64,
    ) -> anyhow::Result<Vec<(String, Vec<service::span::Span>)>> {
        let (mut connection, nodes) = self.connect()?;
        let mut spans = Vec::new();
        let mut id = 0;
        for node in &nodes {
            // Each reply is one page of the node's spans, which fits a message; the read asks
            // for pages until the last.
            let mut read = Vec::new();
            let mut start = Some(since);
            while let Some(since) = start {
                id += 1;
                connection.deliver(&process::Upstream {
                    kind: Some(upstream::Kind::Request(process::Request {
                        id,
                        kind: Some(request::Kind::Spans(process::SpansQuery {
                            host: node.host.clone(),
                            deployment: self.deployment.clone(),
                            since,
                        })),
                    })),
                })?;
                let reply = await_reply(&mut connection, id).context("awaiting a node's spans")?;
                anyhow::ensure!(
                    reply.error.is_empty(),
                    "the spans of {}: {}",
                    node.host,
                    reply.error
                );
                let Some(reply::Kind::Spans(page)) = reply.kind else {
                    anyhow::bail!("the daemon of {} answered its spans with none", node.host);
                };
                let decoded: Vec<service::span::Span> = service::msgpack::decode(&page.spans)
                    .with_context(|| format!("reading the spans of {}", node.host))?;
                read.extend(decoded);
                start = page.next;
            }
            spans.push((node.host.clone(), read));
        }
        Ok(spans)
    }

    /// Has the daemon of `host` prepare the module set `filled` at its own Build Server
    /// (`Request::PrepareModules`), and returns the set as that Build Server prepared it.
    ///
    /// The wait has no deadline: a store that misses the engine's executable builds it for
    /// minutes, and the node's Build Server reports each module's build as it goes. It ends with
    /// the node's reply, which the leader's service gives with a failure when the node's daemon is
    /// lost first, or with this connection's end when this daemon's service ends.
    ///
    /// # Errors
    ///
    /// Fails when the daemon cannot be reached, when the node cannot prepare the set, with its
    /// reason, and when its answer holds no module set.
    pub(crate) fn prepare_on(
        &self,
        host: &str,
        filled: &crate::epoch::ModuleSet,
    ) -> anyhow::Result<crate::epoch::ModuleSet> {
        let modules = Some(filled.into());
        let (mut control, _) = self.connect()?;
        control.deliver(&process::Upstream {
            kind: Some(upstream::Kind::Request(process::Request {
                id: 1,
                kind: Some(request::Kind::PrepareModules(process::PrepareModules {
                    host: host.to_owned(),
                    modules,
                })),
            })),
        })?;
        tracing::info!(
            host,
            "asking a host's daemon to prepare the epoch's modules at its Build Server"
        );
        let reply = await_reply(&mut control, 1)
            .with_context(|| format!("awaiting the preparation at {host}"))?;
        anyhow::ensure!(
            reply.error.is_empty(),
            "preparing the epoch's modules at {host}: {}",
            reply.error
        );
        let Some(reply::Kind::ModulesPrepared(prepared)) = reply.kind else {
            anyhow::bail!("the daemon of {host} answered the preparation with no module set");
        };
        let prepared = prepared
            .modules
            .context("the answer holds no module set")
            .and_then(|modules| Ok(crate::epoch::ModuleSet::try_from(modules)?))
            .with_context(|| format!("reading the module set {host} prepared"))?;
        tracing::info!(host, "a host's daemon prepared the epoch's modules");
        Ok(prepared)
    }

    /// What an epoch update starts the world of one target replica with: the launch file of each of
    /// its engines, numbered `numbers` (the launch generations, the scheduler node's first), with
    /// the reservations it inherits ([`Self::world`]).
    ///
    /// # Errors
    ///
    /// As [`Self::world`].
    pub(crate) fn target_world(
        &self,
        args: &EngineArgs,
        numbers: &[u64],
    ) -> anyhow::Result<Vec<(EngineArgs, Vec<u64>)>> {
        let (launches, inherit) = self.world(args.clone(), numbers)?;
        Ok(launches.into_iter().zip(inherit).collect())
    }

    /// Has the daemon of `launch`'s host start the engine `launch` describes, inheriting the
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
            host == self.host || matches!(self.processes, service::Endpoint::Tcp { .. }),
            "a remote engine needs the deployment's network control address"
        );

        control.deliver(&process::Upstream {
            kind: Some(upstream::Kind::Request(process::Request {
                id: 1,
                kind: Some(request::Kind::Spawn(process::Spawn {
                    host: host.to_owned(),
                    group: ENGINE_GROUP.to_owned(),
                    rank: launch.rank,
                    generation: engine,
                    inherit,
                    // The node that starts the process resolves the engine package its launch
                    // names at its own Build Server and passes it the launch.
                    launch: serde_json::to_vec(launch).context("encoding the launch file")?,
                })),
            })),
        })?;
        let reply = await_reply(&mut control, 1).context("awaiting the engine's start")?;
        anyhow::ensure!(
            reply.error.is_empty(),
            "starting the engine process on {host}: {}",
            reply.error
        );
        if let Some(reply::Kind::Spawned(spawned)) = reply.kind {
            tracing::info!(
                engine,
                host,
                pid = spawned.pid,
                "started the engine process"
            );
        }
        Ok(control)
    }

    /// Accepts engine `engine`'s connection to the request socket, failing when the daemon
    /// reports its process's exit on `control` first.
    ///
    /// The wait blocks on the request socket and on `control`: an engine dials as the first thing
    /// it does, and an engine that exits before is reported on its control connection. Engines
    /// dial in any order; one that dials before it is awaited waits in `incoming` under the number
    /// it states.
    fn accept(
        &self,
        engine: u64,
        mut control: Option<&mut Connection>,
    ) -> anyhow::Result<(Connection, Connection, wire::Hello)> {
        let mut incoming = ::service::lock(&self.incoming);
        if let Some(connection) = incoming.remove(&engine) {
            return Ok(connection);
        }
        // Whether the handoff's control connection has been seen to close: the service
        // goes on with its accepted side, so an end here is a predecessor that left without
        // handing it over, and not the engine. Stated once.
        let mut control_closed = false;
        loop {
            match self.requests.try_accept() {
                Ok(Some(mut reader)) => {
                    let writer = reader.writer()?;
                    // An engine's first message on its request link is its hello.
                    let hello: wire::Hello = reader
                        .recv_next()
                        .context("the engine closed its request link before its hello")?;
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
                    continue;
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
            // A dial or the control connection's report ends the wait.
            let mut descriptors = vec![crate::supervision::readable(self.requests.as_raw_fd())];
            if let Some(control) = control.as_deref()
                && !control_closed
            {
                descriptors.extend(control.poll_fds());
            }
            service::wait_ready(&mut descriptors, None)?;
        }
    }
}

/// Answers the hello `hello` that engine `engine` said on its request link with this daemon's
/// welcome, written on `writer`, the link's write half: the daemon's half of the request
/// interface's handshake, wherever the link was made (this control's request socket, or a pair a
/// test made). The engine is served in the newest version of the engine interface both windows
/// hold ([`proto::ENGINE_INTERFACE`]), which the welcome states; an engine whose window shares
/// none with this daemon's, or one that states another number, is refused in the welcome with the
/// typed refusal. Returns the version agreed on.
///
/// # Errors
///
/// When the welcome cannot be written, or the engine is refused (the refusal, as the error).
pub(crate) fn welcome(
    writer: &mut Connection,
    hello: &wire::Hello,
    engine: u64,
) -> anyhow::Result<u32> {
    let agreed = proto::agree(hello.oldest, hello.newest, proto::ENGINE_INTERFACE)
        .map_err(wire::Refusal::of_versions)
        .and_then(|version| {
            if hello.engine == engine {
                Ok(version)
            } else {
                Err(wire::Refusal {
                    reason: Some(wire::refusal::Reason::Engine(wire::EngineMismatch {
                        stated: hello.engine,
                        expected: engine,
                    })),
                })
            }
        });
    writer.deliver(&wire::Welcome {
        build: proto::BUILD.to_owned(),
        refusal: agreed.as_ref().err().cloned(),
        event_window: crate::engine::client::EVENT_WINDOW,
        version: agreed.as_ref().map_or(0, |version| *version),
    })?;
    agreed.map_err(|refusal| anyhow::Error::new(refusal).context("refused the engine"))
}

/// What the state `state` an engine states on its request link says of its start: the state the
/// start ended in, parked or serving, with what the engine serves; `None` while it starts.
///
/// # Errors
///
/// When the start failed, with the engine's reason, or the engine states no info with the state
/// its start ended in.
pub(crate) fn start_state(state: wire::State) -> anyhow::Result<Option<(EngineInfo, EngineState)>> {
    let ended = match wire::EngineState::try_from(state.state) {
        Ok(wire::EngineState::Parked) => EngineState::Parked,
        Ok(wire::EngineState::Serving) => EngineState::Serving,
        Ok(wire::EngineState::Failed) => {
            anyhow::bail!("the engine failed to start: {}", state.error)
        }
        _ => return Ok(None),
    };
    let info = state.info.context("the engine reported no info")?;
    Ok(Some((info.try_into()?, ended)))
}

/// Reads the request link `reader` of an engine this daemon welcomed until the engine states how
/// its start ended, and returns that: the state, parked or serving, with what the engine serves.
/// The read ends with that state or with the link.
///
/// # Errors
///
/// When the start failed, with the engine's reason; when the link fails or closes first; and as
/// [`start_state`] refuses the state.
pub(crate) fn await_start(reader: &mut Connection) -> anyhow::Result<(EngineInfo, EngineState)> {
    loop {
        let (frame, _) = reader.recv_next_with::<wire::FromEngine>()?;
        if let Some(wire::from_engine::Kind::State(state)) = frame.kind
            && let Some(started) = start_state(state)?
        {
            return Ok(started);
        }
    }
}

/// Reads the reply `id` on a control connection. The daemon's service answers every request on
/// it, so the read ends with the reply or with the connection's end.
/// How a node's KV program started, as [`process::ProgramConfigured`] names it; `None` for the
/// empty name of a node that runs none.
///
/// # Errors
///
/// Fails for a name of no start.
fn program_start(name: &str) -> anyhow::Result<Option<storage::protocol::ProgramStart>> {
    use storage::protocol::ProgramStart;

    match name {
        "" => Ok(None),
        "kept" => Ok(Some(ProgramStart::Kept)),
        "restored" => Ok(Some(ProgramStart::Restored)),
        "created" => Ok(Some(ProgramStart::Created)),
        other => anyhow::bail!("{other:?} names no start of a KV program"),
    }
}

fn await_reply(control: &mut Connection, id: u64) -> anyhow::Result<process::Reply> {
    loop {
        let message: process::Downstream = control.recv_next()?;
        if let Some(downstream::Kind::Reply(reply)) = message.kind
            && reply.id == id
        {
            return Ok(reply);
        }
    }
}

/// Waits until every process of a world reported that it serves, or that it stands by when
/// `standby`, and returns what each reported. The first failure ends the wait: a process that
/// reports its start failed, or exits, as its control connection says. The wait blocks on every
/// process's request link and control connection at once.
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
                        started[index] = Some((info.try_into()?, state));
                        break;
                    }
                    Ok(wire::EngineState::Failed) => {
                        // The failed engine closes its workers and exits;
                        // their storage is free once it has.
                        await_exit(reader, &mut controls[index]);
                        anyhow::bail!("the engine on {host} failed to start: {}", state.error);
                    }
                    _ => {}
                }
            }
        }
        if started.iter().all(Option::is_some) {
            return Ok(started.into_iter().flatten().collect());
        }
        let mut descriptors = Vec::new();
        for (index, control) in controls.iter_mut().enumerate() {
            if started[index].is_none() {
                exited(control)
                    .with_context(|| format!("the engine on {}", launches[index].host))?;
                descriptors.extend(control.poll_fds());
                descriptors.extend(links[index].0.poll_fds());
            }
        }
        // A state on a request link or an exit on a control connection ends the wait.
        service::wait_ready(&mut descriptors, None)?;
    }
}

/// Whether the control connection still reports a serving engine: `Ok(false)` when it has
/// closed, which is a connection ending rather than an engine exit.
fn still_serving(control: &mut Connection) -> anyhow::Result<bool> {
    match exited(control) {
        Ok(()) => Ok(true),
        Err(error) => match error.downcast_ref::<std::io::Error>() {
            Some(closed) if closed.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
            _ => Err(error),
        },
    }
}

/// Waits until the engine process that reported its failure exited: the daemon reports the exit
/// on its control connection `control`, and the process's end closes its request link `reader`.
/// Whichever comes first ends the wait.
fn await_exit(reader: &mut Connection, control: &mut Connection) {
    loop {
        if exited(control).is_err() {
            return;
        }
        // A process that goes closes this connection, and the daemon then forgets it without
        // reporting an exit on `control`.
        loop {
            match reader.try_recv::<wire::FromEngine>() {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(_) => return,
            }
        }
        let mut descriptors = control.poll_fds().to_vec();
        descriptors.extend(reader.poll_fds());
        if service::wait_ready(&mut descriptors, None).is_err() {
            return;
        }
    }
}

/// Fails when the daemon reported the engine process's exit on its control
/// connection, with its exit status and the error it stated as it ended (its exit reason).
fn exited(control: &mut Connection) -> anyhow::Result<()> {
    while let Some(message) = control.try_recv::<process::Downstream>()? {
        if let Some(downstream::Kind::Event(process::Event {
            kind: Some(event::Kind::Exit(exit)),
        })) = message.kind
        {
            if exit.reason.is_empty() {
                anyhow::bail!("the engine process exited ({})", exit.status);
            }
            anyhow::bail!(
                "the engine process exited ({}): {}",
                exit.status,
                exit.reason
            );
        }
    }
    Ok(())
}

/// One host's Build Server, as the daemon of that host reaches it, and the targets that host's
/// devices build for.
#[derive(Debug, Clone)]
pub struct BuildServer {
    /// The host's identity in the deployment.
    pub host: String,
    /// Its Build Server's socket, on that host.
    pub socket: PathBuf,
    /// The targets its devices compile for, which `prepare` builds the epoch's modules for; empty
    /// for the Build Server's own: no daemon queries another host's devices to name them.
    pub targets: Vec<build_server::Target>,
}

/// Prepares the module set `filled`, whose revisions are fixed ([`with_store_modules`]), at the
/// Build Server of every host in `hosts`, and returns it with the parameters each package runs
/// with. The daemon calls it before it spawns any process of the epoch, at a start and at every
/// apply, so that the main process a node later starts reads its engine's module from that node's
/// store and a spawn builds nothing.
///
/// `local` is this daemon's own host's Build Server, which it reaches at its socket and which
/// prepares in any case: its answer is the set the epoch records. Every other host of `hosts`
/// prepares at its own Build Server through its daemon (`Request::PrepareModules`), which `nodes`, this
/// daemon's engine control, reaches over the process socket. The hosts prepare at once, each for
/// its own platform and devices, so the set is ready when the slowest host's store is. The same
/// revision names the same module on every host, so a host whose Build Server lacks one fails the
/// preparation rather than choosing another.
///
/// Every wait ends with a reply or a connection's end: a Build Server answers a prepare once it
/// built every module, reporting each build as it goes, and another host's daemon answers once
/// its Build Server did, or the leader's service fails the request with that daemon's loss.
///
/// # Errors
///
/// Fails when a Build Server refuses or cannot build a module, when no daemon runs on a host of
/// `hosts` or it is lost before it answers, when another host is named and no `nodes` reaches it,
/// and when a host prepares other modules than this host's.
pub fn prepare_epoch(
    local: &BuildServer,
    nodes: Option<&EngineControl>,
    hosts: &[String],
    filled: &crate::epoch::ModuleSet,
) -> anyhow::Result<crate::epoch::ModuleSet> {
    let mut others: Vec<&str> = Vec::new();
    for host in hosts {
        if *host != local.host && !others.contains(&host.as_str()) {
            others.push(host);
        }
    }
    tracing::info!(
        host = %local.host,
        others = ?others,
        kernels = filled.kernels.len(),
        policy = %filled.policy.backend,
        "preparing the epoch's modules at every host's Build Server"
    );

    // Every other host prepares on a thread of its own while this one prepares here: a cold store
    // builds the engine's executable for minutes on each host, and the hosts do not wait on one
    // another.
    let (pinned, remote) = std::thread::scope(|scope| {
        let waits: Vec<_> = others
            .iter()
            .map(|&host| {
                let wait = scope.spawn(move || {
                    let nodes = nodes.with_context(|| {
                        format!("no daemon of the deployment reaches host {host}")
                    })?;
                    nodes.prepare_on(host, filled)
                });
                (host, wait)
            })
            .collect();
        let pinned = prepare_at(local, filled);
        let remote: Vec<_> = waits
            .into_iter()
            .map(|(host, wait)| (host, wait.join()))
            .collect();
        (pinned, remote)
    });
    let pinned = pinned?;
    for (host, prepared) in remote {
        let mut prepared =
            prepared.map_err(|_| anyhow::anyhow!("the preparation at {host} panicked"))??;
        // The engines' wire form of the set leaves out what the leader daemon runs itself (its
        // programs and its processing program), which no other host prepares.
        prepared.programs.clone_from(&pinned.programs);
        prepared.processing.clone_from(&pinned.processing);
        anyhow::ensure!(
            prepared == pinned,
            "the Build Server of {host} prepared other modules than the Build Server of {}",
            local.host
        );
    }
    tracing::info!(
        hosts = others.len() + 1,
        kernels = pinned.kernels.len(),
        policy = %pinned.policy.backend,
        "the epoch's modules are prepared at every host"
    );
    Ok(pinned)
}

/// Prepares the module set a node request names (`Request::PrepareModules`) at this node's Build Server,
/// which `settings` states and which is started when none serves there, and returns the set as
/// that Build Server prepared it. A node's daemon runs it for the leader, off its service.
///
/// # Errors
///
/// The reason the reply states: the request's set does not read, this node's daemon states no
/// Build Server, or the Build Server cannot be started or refuses or cannot build a module.
pub(crate) fn prepare_node(
    runtime: &service::RuntimeDirectory,
    deployment: &str,
    settings: Option<&build_server::command::Settings>,
    prepare: &process::PrepareModules,
) -> Result<process::ModulesPrepared, String> {
    let filled = prepare
        .modules
        .clone()
        .ok_or_else(|| "the request names no module set to prepare".to_owned())
        .and_then(|modules| {
            crate::epoch::ModuleSet::try_from(modules)
                .map_err(|error| format!("reading the module set to prepare: {error}"))
        })?;
    let settings =
        settings.ok_or_else(|| "this node's daemon states no Build Server".to_owned())?;
    let socket = ensure_build_server(runtime, deployment, settings)
        .map_err(|error| format!("starting this node's Build Server: {error:#}"))?;
    let server = BuildServer {
        host: prepare.host.clone(),
        socket,
        targets: Vec::new(),
    };
    let prepared = prepare_at(&server, &filled).map_err(|error| format!("{error:#}"))?;
    Ok(process::ModulesPrepared {
        modules: Some((&prepared).into()),
    })
}

/// `filled` as a fixed module set: every package it names at the revision `server`, the leader
/// daemon's own host's Build Server, selects for it, and every installed kernel package of every
/// operator of the fixed engine revision at its selected revision.
///
/// This is the one selection of an apply. The daemon asks `cover` for each interface the
/// deployment uses: the engine, the scheduling policy and every kernel operator the fixed engine
/// revision launches, as the Build Server lists that revision's vocabulary ([`engine_operators`])
/// rather than as this daemon's build knows operators, so the packages of an operator a later
/// engine revision adds are fixed, pinned and moved by a newer revision like any other's. Each
/// operator is asked at the version of its interface the vocabulary states, so only the package
/// revisions that implement it are candidates: a revision of another version, installed for
/// another engine revision, would be refused by this revision's engines when they load it, and
/// is fixed for no epoch of this one. An engine revision whose vocabulary is of format 1
/// (`build_server::manifest::VOCABULARY_FORMATS`) states no versions, and its operators are
/// asked at none, so their candidates are each backend's latest revisions as that format's
/// were. A package the deployment names, the engine, the policy or a
/// pin of its `[kernels]` table, takes the revision it states, else its latest installed one (of
/// the operator's version, for a pin); a pin of an operator the engine revision does not launch,
/// or of a revision of another version, is refused. Every candidate of an operator is
/// recorded as well, one slot per package with the package as its part, a pinned backend at its
/// pinned revision: for each descriptor key the engine later asks `cover` to select among exactly
/// these revisions, so a package or a revision installed after this call is run by nobody until
/// a configuration is applied again, and a later revision of one of them is a package change a
/// configuration comparison sees. Every other host is then asked for exactly these revisions
/// ([`prepare_epoch`]).
///
/// # Errors
///
/// Fails when the Build Server cannot be reached, or holds no package, or not the revision, the
/// deployment names for an interface, or refuses the pins of an operator, and when the
/// deployment pins an operator the engine revision does not launch or a revision of another
/// version of the operator's interface than the engine revision launches.
pub fn with_store_modules(
    server: &BuildServer,
    filled: &crate::epoch::ModuleSet,
) -> anyhow::Result<crate::epoch::ModuleSet> {
    let mut client = build_server::Client::connect(&server.socket).with_context(|| {
        format!(
            "connecting to the Build Server of {} at {}",
            server.host,
            server.socket.display()
        )
    })?;

    let mut complete = filled.clone();
    // The engine and the programs state the versions of their interfaces in their modules, which
    // their loaders check; their manifests state none to select by.
    fix_revision(
        &mut client,
        proto::module::ENGINE,
        None,
        &mut complete.engine,
    )?;
    let operators = engine_operators(&mut client, &complete.engine)?;
    fix_revision(
        &mut client,
        proto::module::PROGRAM,
        None,
        &mut complete.policy,
    )?;
    if let Some(processing) = complete.processing.as_mut() {
        fix_revision(&mut client, proto::module::PROGRAM, None, processing)?;
    }
    let engine = &complete.engine;
    for (slot, package) in &mut complete.kernels {
        let operator = operators
            .iter()
            .find(|operator| operator.name == slot.op)
            .with_context(|| {
                format!(
                    "`kernels.{}` names no operator engine {} revision {} launches",
                    slot.op, engine.backend, engine.revision
                )
            })?;
        fix_revision(&mut client, &slot.op, operator.interface_version, package).with_context(
            || match operator.interface_version {
                Some(version) => format!(
                    "`kernels.{}` of engine {} revision {}, which launches version {version} of \
                     the operator's interface",
                    slot.op, engine.backend, engine.revision
                ),
                None => format!(
                    "`kernels.{}` of engine {} revision {}",
                    slot.op, engine.backend, engine.revision
                ),
            },
        )?;
    }
    for operator in &operators {
        // The operator's pins, at their fixed revisions: a pinned backend is the operator's
        // candidate at the revision its pins state.
        let pins = complete
            .kernels
            .iter()
            .filter(|(slot, _)| slot.op == operator.name)
            .filter_map(|(slot, package)| {
                let pin = proto::module::Pin {
                    backend: package.backend.clone(),
                    revision: Some(package.revision),
                };
                Some((slot.key.clone()?, pin))
            })
            .collect();
        let selected = candidates(
            &mut client,
            &operator.name,
            operator.interface_version,
            pins,
            Vec::new(),
        )?;
        for candidate in selected {
            complete.kernels.insert(
                proto::module::ModuleSlot {
                    op: candidate.op,
                    key: None,
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

/// The operators whose kernel packages the engine package revision `engine` launches, each with
/// the version of its interface the revision launches, as the Build Server lists them with the
/// revision: the vocabulary of the revision's own snapshot (`operators` of its manifest), which
/// names every operator its executable was compiled with and the version it launches, whether
/// or not this daemon's build knows the operator.
///
/// # Errors
///
/// Fails when the Build Server does not list its packages, or lists no such engine revision.
fn engine_operators(
    client: &mut build_server::Client,
    engine: &proto::module::PackageRevision,
) -> anyhow::Result<Vec<build_server::manifest::Operator>> {
    client
        .list()
        .context("listing the Build Server's packages")?
        .into_iter()
        .find(|entry| {
            entry.package.op == proto::module::ENGINE
                && entry.package.backend == engine.backend
                && entry.package.revision == engine.revision
        })
        .map(|entry| entry.operators)
        .with_context(|| {
            format!(
                "the Build Server lists no engine package {} revision {}",
                engine.backend, engine.revision
            )
        })
}

/// The candidates the Build Server selects among for `op`: every installed package of the
/// interface that `fixed` leaves a candidate, each at its latest revision, the revision `pins`
/// state of a pinned backend, or the revision `fixed` states, of the interface version
/// `interface_version` when it states one.
fn candidates(
    client: &mut build_server::Client,
    op: &str,
    interface_version: Option<u32>,
    pins: std::collections::BTreeMap<String, proto::module::Pin>,
    fixed: Vec<(String, u32)>,
) -> anyhow::Result<Vec<build_server::protocol::Package>> {
    let coverage = client
        .cover(&build_server::CoverRequest {
            op: op.to_owned(),
            // No target: a module set fixes revisions for every host of the deployment, and each
            // engine selects among them for its own devices.
            targets: Vec::new(),
            keys: Vec::new(),
            pins,
            fixed,
            interface_version,
        })
        .with_context(|| format!("asking the Build Server for the packages of {op}"))?;
    Ok(coverage.candidates)
}

/// Fixes the revision of the package `package` names for `op`: the one it states, checked, or
/// its latest installed one; of the version `interface_version` of the operator's interface
/// when it states one, so a stated revision of another version is refused.
pub(crate) fn fix_revision(
    client: &mut build_server::Client,
    op: &str,
    interface_version: Option<u32>,
    package: &mut proto::module::PackageRevision,
) -> anyhow::Result<()> {
    let stated = (package.revision != 0).then(|| (package.backend.clone(), package.revision));
    let selected = candidates(
        client,
        op,
        interface_version,
        Default::default(),
        stated.into_iter().collect(),
    )?
    .into_iter()
    .find(|candidate| candidate.backend == package.backend)
    .with_context(|| match interface_version {
        Some(version) => format!(
            "no {op} package {} implementing version {version} of the operator's interface is \
             installed",
            package.backend
        ),
        None => format!("no {op} package {} is installed", package.backend),
    })?;
    package.revision = selected.revision;
    Ok(())
}

/// Runs `prepare` at one host's Build Server, which this process reaches at its socket, for the
/// module set `filled`, whose revisions are fixed, and returns it with the parameters each
/// package runs with: its manifest's defaults under the deployment's arguments.
fn prepare_at(
    server: &BuildServer,
    filled: &crate::epoch::ModuleSet,
) -> anyhow::Result<crate::epoch::ModuleSet> {
    let policy_slot = proto::module::ModuleSlot::all(proto::module::PROGRAM);
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
    // starts an engine from it: a host whose store misses it builds the
    // executable for its own platform now, while nothing waits on a start.
    // The kernels' modules are built for the keys each engine's profile runs
    // name, which an engine resolves when it starts.
    let request = build_server::PrepareRequest {
        targets: server.targets.clone(),
        packages: requests,
        descriptors: vec![build_server::Descriptors {
            slot: engine_slot.clone(),
            keys: vec![proto::module::SINGLE_KEY.to_owned()],
        }],
    };
    // Before the connect, not after it: a daemon that blocks opening the
    // socket names the host it was reaching from this line alone, and the
    // server it never reached logs nothing at all.
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
    let prepared = client
        .prepare(&request, &mut progress)
        .with_context(|| format!("preparing the epoch's modules at {}", server.host))?;
    tracing::info!(
        host = %server.host,
        prepared = prepared.len(),
        "a Build Server responded"
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
        // The programs were fixed before the set: the Build Servers prepare the engines' modules,
        // and the leader daemon resolves the processing program itself.
        programs: filled.programs.clone(),
        processing: filled.processing.clone(),
    })
}

/// One engine process on a node, by the engine number its process link is
/// registered under, which a command names and a report carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpochProcess {
    /// The node the engine runs on.
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

/// What a node's engine reaches, as its launch file states it.
pub(crate) struct Interfaces<'a> {
    /// The deployment the process serves, by its name.
    pub(crate) deployment: &'a str,
    /// The leader daemon's request interface.
    pub(crate) requests: &'a str,
    /// The daemon's process interface.
    pub(crate) processes: &'a str,
    /// The Build Server's socket.
    pub(crate) build_server: &'a Path,
}

/// The launch files of the world of `args`, one engine per host of its worker groups in
/// [`world_hosts`]'s order (the scheduler node's first), numbered `numbers`: each process with its
/// host's place in a world of every rank of the deployment ([`world_size`]), all in the epoch of
/// the first. Where the processes meet is stated by the start that reserves the world's sockets.
///
/// # Errors
///
/// Fails when `numbers` does not number every host, and as [`world_hosts`] and [`place_on`] fail.
fn place_world(
    args: EngineArgs,
    numbers: &[u64],
    interfaces: &Interfaces<'_>,
) -> anyhow::Result<Vec<EngineArgs>> {
    let hosts = world_hosts(&args.engine_config)?;
    anyhow::ensure!(
        numbers.len() == hosts.len(),
        "{} engine numbers for the engines of {} hosts",
        numbers.len(),
        hosts.len()
    );
    let world_size = world_size(&args.engine_config)?;
    let mut launches = hosts
        .iter()
        .zip(numbers)
        .map(|(host, number)| place_on(args.clone(), *number, interfaces, host, world_size))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // A new epoch takes its first engine's number, on every host.
    let epoch = launches[0].epoch;
    for launch in &mut launches {
        launch.epoch = epoch;
    }
    Ok(launches)
}

/// The ranks of the deployment's worker groups one host holds: an engine's place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HostRanks {
    /// The host, by host identity.
    pub(crate) host: String,
    /// The deployment rank of the first rank it holds, in the numbering of every worker group's
    /// ranks in the order the groups are stated: the rank its launch states and its process is
    /// supervised under. The host's other ranks need not follow it, since each group of a
    /// deployment may place ranks on several hosts.
    pub(crate) first: u32,
    /// The CUDA ordinals of its ranks, in rank order, each once: a device may hold ranks of
    /// several groups.
    pub(crate) devices: Vec<u32>,
}

/// The Storage Server scopes of `config`'s ranks on `host`: each rank's worker
/// group and its number in the deployment, its place among the ranks of the
/// deployment's groups in their order
/// ([`proto::config::WorkerGroupConfig::first_ranks`]), as the engine numbers
/// its ranks. A rank's workers retain their tensors under this scope, and a
/// claim of the epoch states the scopes it places on the node: one that leaves
/// a scope out ends the epoch's retention of it.
pub(crate) fn host_scopes(
    config: &proto::config::EngineConfig,
    host: &str,
) -> Vec<proto::process::Scope> {
    let groups = &config.parallel.worker_groups;
    let firsts = proto::config::WorkerGroupConfig::first_ranks(groups);
    let mut scopes = Vec::new();
    for (first, group) in firsts.into_iter().zip(groups) {
        for (position, rank) in group.ranks.iter().enumerate() {
            if rank.host != host {
                continue;
            }
            if let Ok(number) = u32::try_from(first + position) {
                scopes.push(proto::process::Scope {
                    group: group.id.0.clone(),
                    rank: number,
                });
            }
        }
    }
    scopes.sort_unstable_by_key(|scope| scope.rank);
    scopes
}

/// The hosts of the deployment's worker groups, in the order of the first rank each holds, each
/// with that rank and the devices of its ranks: one engine per host serves an epoch
/// (`docs/architecture.md`, The channel), the engine host's first, holding rank 0 and the
/// schedulers.
///
/// A host's ranks may lie anywhere in the deployment's numbering: a deployment that places one
/// replica of each of two groups on each of two hosts numbers the first group's ranks of both
/// hosts before the second group's.
///
/// # Errors
///
/// Fails when the engine host does not hold rank 0, or as [`local_devices`] fails.
pub(crate) fn world_hosts(config: &proto::config::EngineConfig) -> anyhow::Result<Vec<HostRanks>> {
    let engine_host = config.parallel.engine_host.as_str();
    // Each host with the deployment rank of the first rank it holds.
    let firsts = proto::config::WorkerGroupConfig::hosts(&config.parallel.worker_groups);
    let first_host = firsts.first().map(|(host, _)| *host);
    anyhow::ensure!(
        first_host == Some(engine_host),
        "the deployment's first rank runs on {first_host:?} and its engine host is {engine_host}: \
         the engine host's engine holds rank 0 and the schedulers"
    );

    firsts
        .into_iter()
        .map(|(host, first)| {
            Ok(HostRanks {
                host: host.to_owned(),
                first: u32::try_from(first)
                    .context("a deployment of more ranks than u32 numbers")?,
                devices: local_devices(host, config)?,
            })
        })
        .collect()
}

/// The number of the deployment's ranks, every worker group's on every host: the members of the
/// host group its engines' ranks meet in, each under its deployment rank. A device that holds
/// ranks of several groups counts once for each.
///
/// # Errors
///
/// Fails for a deployment of more ranks than u32 numbers.
fn world_size(config: &proto::config::EngineConfig) -> anyhow::Result<u32> {
    let ranks: usize = config
        .parallel
        .worker_groups
        .iter()
        .map(|group| group.ranks.len())
        .sum();
    u32::try_from(ranks).context("a deployment of more ranks than u32 numbers")
}

/// The launch file's contents for the process `engine` on `host`, in a world of `world_size`
/// ranks: `args`, with the process's place and the interfaces `interfaces` states. The node that
/// starts it states its own Build Server (`Spawn::launch`).
///
/// A value the file requires and the daemon cannot state is an error here rather than a default:
/// the engine refuses a launch file that lacks one, and a default would be a fact the daemon
/// invented about the process.
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
    let scheduler_node = host.host == config.parallel.engine_host;
    // A new epoch takes the identity of its first engine; an engine that
    // replaces a lost one goes on with its epoch.
    if args.epoch == 0 {
        args.epoch = engine;
    }
    args.world_size = world_size;
    args.deployment = interfaces.deployment.to_owned();
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
    Ok(args)
}

/// The CUDA ordinals `host`'s ranks serve, in rank order and without repeats.
///
/// A deployment that states no `device_config.devices` still gives each
/// host's ranks the ordinals the shorthand numbers them from zero; one that
/// does state an order has already had it folded into the ranks by
/// [`proto::config::WorkerGroupConfig::shorthand`]. Reading the ranks is the
/// one account of both, and it is what the launch file states as the devices
/// of this node's process.
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
            let ordinal = rank.device.cuda_ordinal().with_context(|| {
                format!(
                    "host {host} runs a rank on {}: an engine serves CUDA devices",
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
    use crate::config::Config;

    /// A launch places the scheduler node's process on the engine host's devices, as the
    /// deployment's ranks name them, with the interfaces the leader states and none it does not.
    #[test]
    fn a_launch_places_the_process_on_the_engine_hosts_devices() {
        let mut options = Config {
            model: Some("model".to_owned()),
            hosts: vec!["rank-0".to_owned()],
            tensor_parallel_size: std::num::NonZeroUsize::new(2).unwrap(),
            ..Default::default()
        };
        options.device_config.devices = vec![6, 7];
        let args = EngineArgs::new(
            options.engine_config().expect("the configuration resolves"),
            crate::epoch::modules_of(&options),
        );
        let interfaces = Interfaces {
            deployment: "",
            requests: "",
            processes: "",
            build_server: Path::new(""),
        };

        let [placed] = place_world(args, &[1], &interfaces)
            .expect("the process places")
            .try_into()
            .expect("one host, one process");
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

    /// A worker group whose ranks spread over two hosts has one engine per host, the engine
    /// host's first and holding rank 0, each with its own run of the group's ranks.
    #[test]
    fn a_group_over_two_hosts_is_one_process_per_host() {
        let mut options = Config {
            model: Some("model".to_owned()),
            hosts: vec!["rank-0".to_owned()],
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
        crate::epoch::modules_of(&Config::default())
    }

    /// A worker group `id` of `role` whose ranks are `ranks`, each a host and a GPU of it, in rank
    /// order.
    fn group(
        id: &str,
        role: proto::config::WorkerGroupRole,
        ranks: &[(&str, u32)],
    ) -> proto::config::WorkerGroupConfig {
        proto::config::WorkerGroupConfig {
            id: proto::config::WorkerGroupId(id.to_owned()),
            role,
            copies: 1,
            ranks: ranks
                .iter()
                .map(|&(host, device)| proto::config::WorkerRank {
                    host: host.to_owned(),
                    device: proto::config::DeviceKind::Cuda.rank_device(device),
                })
                .collect(),
            blocks: BTreeMap::new(),
            queue_depth: 2,
            mem_fraction_static: None,
        }
    }

    /// The launches of a deployment of the worker groups `groups`, whose hosts are `rank-0`, the
    /// engine host, and `rank-1`: one engine per host, numbered 1 and 2, with each launch's host,
    /// first rank, devices, world size, whether it is the scheduler node, and epoch.
    fn two_host_launches(
        groups: Vec<proto::config::WorkerGroupConfig>,
    ) -> Vec<(String, u32, Vec<u32>, u32, bool, u64)> {
        let options = Config {
            model: Some("model".to_owned()),
            hosts: vec!["rank-0".to_owned()],
            ..Default::default()
        };
        let mut config = options.engine_config().expect("the configuration resolves");
        config.parallel.worker_groups = groups;
        let args = EngineArgs::new(config, modules());
        let interfaces = Interfaces {
            deployment: "",
            requests: "",
            processes: "",
            build_server: Path::new(""),
        };

        place_world(args, &[1, 2], &interfaces)
            .expect("the world places")
            .into_iter()
            .map(|launch| {
                (
                    launch.host,
                    launch.rank,
                    launch.devices,
                    launch.world_size,
                    launch.scheduler_node,
                    launch.epoch,
                )
            })
            .collect()
    }

    /// Each host of a two-host deployment runs one engine from the first deployment rank it holds,
    /// on the devices of its ranks, in a world of every rank of the deployment; the engine host's
    /// is the scheduler node, and every engine is in the epoch of the first. A host's ranks may lie
    /// on either side of another host's.
    #[test]
    fn each_host_of_a_two_host_deployment_starts_from_its_first_rank_in_a_world_of_every_rank() {
        use proto::config::WorkerGroupRole::{Experts, Model};

        // Two groups, each with a replica on each host's four devices: the first group's ranks
        // are 0-3 (`rank-0`) and 4-7 (`rank-1`), the second group's ranks 8-11 and 12-15.
        let spread = |id| {
            let ranks: Vec<(&str, u32)> = ["rank-0", "rank-1"]
                .into_iter()
                .flat_map(|host| (0..4).map(move |device| (host, device)))
                .collect();
            group(id, Model, &ranks)
        };
        assert_eq!(
            two_host_launches(vec![spread("first"), spread("second")]),
            vec![
                ("rank-0".to_owned(), 0, vec![0, 1, 2, 3], 16, true, 1),
                ("rank-1".to_owned(), 4, vec![0, 1, 2, 3], 16, false, 1),
            ]
        );

        // A model group on `rank-0` (ranks 0 and 1) and its expert group on `rank-1` (ranks 2
        // and 3).
        assert_eq!(
            two_host_launches(vec![
                group("model", Model, &[("rank-0", 0), ("rank-0", 1)]),
                group("experts", Experts, &[("rank-1", 0), ("rank-1", 1)]),
            ]),
            vec![
                ("rank-0".to_owned(), 0, vec![0, 1], 4, true, 1),
                ("rank-1".to_owned(), 2, vec![0, 1], 4, false, 1),
            ]
        );
    }

    /// An epoch update's target starts from a launch that states its own launch generation, the
    /// epoch, the daemon's interfaces and the epoch's module set; one that names the engine it
    /// succeeds starts parked, and one that names none serves.
    #[test]
    fn a_targets_launch_states_its_generation_its_modules_and_whether_it_parks() {
        let directory = tempfile::tempdir().unwrap();
        let control = EngineControl::new(
            &directory.path().join("requests.sock"),
            service::Endpoint::unix(directory.path().join("processes.sock")).to_string(),
            "rank-0".into(),
            "model".into(),
            directory.path().join("build-server.sock"),
        )
        .unwrap();
        let mut options = Config {
            model: Some("model".to_owned()),
            served_model_name: Some("model".to_owned()),
            hosts: vec!["rank-0".to_owned()],
            ..Default::default()
        };
        options.device_config.devices = vec![0];
        let mut args = EngineArgs::new(
            options.engine_config().expect("the configuration resolves"),
            modules(),
        );
        args.epoch = 3;

        let world = control.target_world(&args, &[9]).unwrap();
        let (launch, inherit) = &world[0];
        assert_eq!(world.len(), 1);
        assert!(inherit.is_empty(), "a world of one host reserves nothing");
        assert_eq!(launch.engine, 9);
        assert_eq!(launch.epoch, 3);
        assert_eq!(
            launch.processes,
            service::Endpoint::unix(directory.path().join("processes.sock")).to_string()
        );
        assert_eq!(launch.requests, control.request_address());
        assert_eq!(launch.world_size, 1);
        assert_eq!(launch.modules, modules());
        assert!(!launch.parked());

        args.predecessor = Some(wire::Predecessor { epoch: 3 });
        let parked = control.target_world(&args, &[10]).unwrap();
        assert!(parked[0].0.parked());
        assert_eq!(parked[0].0.epoch, 3);
        assert!(
            control.target_world(&args, &[11, 12]).is_err(),
            "one number for each host of the world"
        );
    }

    /// Every independent replica of an epoch is launched in the epoch's one number, whichever
    /// engine number its process gets: the replicas' storages, claims and buffer pool sessions
    /// are the epoch's, and the next epoch's number lies above all of them.
    #[test]
    fn every_replica_of_an_epoch_is_launched_in_its_number() {
        let directory = tempfile::tempdir().unwrap();
        let control = EngineControl::new(
            &directory.path().join("requests.sock"),
            service::Endpoint::unix(directory.path().join("processes.sock")).to_string(),
            "rank-0".into(),
            "model".into(),
            directory.path().join("build-server.sock"),
        )
        .unwrap();
        let mut options = Config {
            model: Some("model".to_owned()),
            served_model_name: Some("model".to_owned()),
            hosts: vec!["rank-0".to_owned()],
            data_parallel_size: std::num::NonZeroUsize::new(2).unwrap(),
            ..Default::default()
        };
        options.device_config.devices = vec![0, 1];
        let config = options.engine_config().expect("the configuration resolves");

        // The deployment's first epoch is numbered as its first engine is.
        let epoch = control.next_engine();
        let launches = crate::leader::epoch_launches(&config, &modules(), epoch, 0).unwrap();
        assert_eq!(launches.len(), 2, "two independent replicas");
        let placed: Vec<EngineArgs> = launches
            .iter()
            .zip([epoch, epoch + 1])
            .map(|(args, engine)| control.target_world(args, &[engine]).unwrap()[0].0.clone())
            .collect();
        let epochs: Vec<u64> = placed.iter().map(|launch| launch.epoch).collect();
        assert_eq!(epochs, vec![epoch, epoch]);
        let groups: Vec<Vec<&str>> = placed
            .iter()
            .map(|launch| {
                let groups = &launch.engine_config.parallel.worker_groups;
                groups.iter().map(|group| group.id.0.as_str()).collect()
            })
            .collect();
        assert_eq!(groups, vec![vec!["model-0"], vec!["model-1"]]);
    }

    /// A handoff's control connection whose accepted side is gone closed with the
    /// predecessor's service, and that is the predecessor leaving, not an engine exit:
    /// the successor waits for the engine's own dial instead of refusing the adoption.
    #[test]
    fn a_closed_control_connection_is_not_an_engine_exit() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("processes.sock");
        let endpoint = service::Endpoint::unix(&path);
        let listener = Listener::bind(&endpoint).unwrap();
        // The predecessor's client end, whose accepted side closed with its service.
        let mut control = Connection::connect(&endpoint).unwrap();
        drop(listener.try_accept().expect("the accepted connection"));
        drop(listener);

        // The close reaches this end once no process holds the accepted side's socket, and a
        // child another thread of this process is spawning holds a copy until it starts its
        // program: wait for the close, as the adopting daemon's wait does, rather than look once.
        let closed = control.recv_next::<process::Downstream>();
        assert_eq!(
            closed.map(drop).map_err(|error| error.kind()),
            Err(std::io::ErrorKind::UnexpectedEof)
        );
        assert!(
            !still_serving(&mut control).expect("a closed connection is not an engine exit"),
            "a closed control connection reports that the daemon's own client is gone"
        );
    }
}
