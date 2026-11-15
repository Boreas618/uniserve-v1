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
use proto::process::{self, downstream, event, hello, reply, request, upstream};
use proto::request::wire;
use service::{Connection, Listener};

use crate::engine::client::{EngineClient, EngineInfo, EngineState, StatsSink};

/// The worker group a daemon supervises an engine process under.
pub(crate) const ENGINE_GROUP: &str = "engine";

/// Makes sure this host's Build Server serves on the socket `settings` names, and returns that
/// socket: when none serves, the deployment's daemon starts it with `settings` as a host service
/// of the running build ([`crate::supervision::RUNNING_IMAGE`], `service::host::start`), which
/// outlives this daemon. The start ends with the server's readiness record, or with the server's
/// end before it served.
///
/// # Errors
///
/// Fails when no Build Server serves and none can be started.
pub fn ensure_build_server(settings: &build_server::command::Settings) -> anyhow::Result<PathBuf> {
    let socket = settings
        .socket()
        .context("finding this host's Build Server socket")?;
    ::service::host::start::<()>(
        Path::new(crate::supervision::RUNNING_IMAGE),
        "build-server",
        &socket,
        &settings.arguments(),
        None,
    )
    .context("starting this host's Build Server")?;
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
        let engine = self.next_engine.fetch_add(1, Ordering::Relaxed);
        self.start_at(args, engine, stats)
    }

    /// [`EngineControl::start`] under a number the caller allocated.
    ///
    /// A standby's number comes from the epoch records' one allocator
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
            requests: &self.request_address,
            processes: &processes,
            build_server: &self.build_server,
        };
        let mut launches = place_world(args, numbers, &interfaces)?;
        for number in numbers {
            self.next_engine.fetch_max(number + 1, Ordering::Relaxed);
        }

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
    /// two ends.
    fn welcome(
        &self,
        engine: u64,
        control: &mut Connection,
    ) -> anyhow::Result<(Connection, Connection)> {
        let (reader, mut writer, hello) = self.accept(engine, Some(control))?;
        welcome(&mut writer, &hello, engine)?;
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

    /// Joins an epoch update's target world, which executes `epoch`: every engine of `processes`
    /// (by host and launch generation, the scheduler node's first) the update started dials this
    /// daemon's request socket, is welcomed, and states that it parked. The update commands them
    /// on their process links, so these connections carry only requests once they serve; the
    /// scheduler node's client holds its peers'. `stats` receives its schedulers' statistics.
    ///
    /// # Errors
    ///
    /// Fails when an engine does not connect in time, speaks a major version this daemon does not,
    /// fails to start, or exits before it stated that it parked.
    pub fn join_target(
        &self,
        processes: &[(String, u64)],
        epoch: u64,
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
        let mut clients = Vec::with_capacity(processes.len());
        let mut sinks = std::iter::once(stats);
        for (host, engine) in processes {
            self.next_engine.fetch_max(engine + 1, Ordering::Relaxed);
            // A peer holds no scheduler, so it reports no statistics.
            let sink = sinks.next().unwrap_or_else(|| Box::new(|_, _, _| {}));
            clients.push(self.join(*engine, epoch, sink)?.on_host(host));
        }
        let mut clients = clients.into_iter();
        let scheduler = clients.next().context("a target of no engine")?;
        Ok(scheduler.with_peers(clients.collect()))
    }

    /// Joins the one target engine `engine` of [`Self::join_target`]. The start has no bound: an
    /// engine loads its ranks, resolves its modules and captures its graphs before it states that
    /// it parked, and one that exits meanwhile closes its request link, which ends the wait.
    fn join(&self, engine: u64, epoch: u64, stats: StatsSink) -> anyhow::Result<EngineClient> {
        let (mut reader, mut writer, hello) = self.accept(engine, None)?;
        welcome(&mut writer, &hello, engine)?;
        loop {
            let message: wire::FromEngine = reader.recv_next()?;
            let Some(wire::from_engine::Kind::State(state)) = message.kind else {
                continue;
            };
            let Some((info, known)) = start_state(state)? else {
                continue;
            };
            tracing::info!(engine, build = %hello.build, "the update's target joined this daemon");
            return EngineClient::new(reader, writer, info, known, stats, None)
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
    /// Fails when the engine's process exits or its request link closes before it states its
    /// state, when it speaks another version of the engine interface than this daemon, is another
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
        welcome(&mut writer, &hello, engine)?;
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
                tracing::info!(engine, build = %hello.build, "the engine attached");
                // A daemon adopts the engine of its own node, which runs on this host: the
                // updater names the processes it drains and commands by their host
                // (`EngineClient::processes`), as it does for an engine this control started.
                return EngineClient::new(
                    reader,
                    writer,
                    info.into(),
                    EngineState::Serving,
                    stats,
                    control,
                )
                .map(|client| client.numbered(engine).in_epoch(epoch).on_host(&self.host));
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

    /// Publishes one commit decision at every deployment node. Publication
    /// is atomic within each Storage Server and idempotent across retries;
    /// the daemon retains its commit decision until every node has replied.
    pub fn publish_epochs(
        &self,
        epochs: &[u64],
        pending: &mut Option<Vec<String>>,
    ) -> anyhow::Result<()> {
        let (mut connection, nodes) = self.connect()?;
        let pending =
            pending.get_or_insert_with(|| nodes.into_iter().map(|node| node.host).collect());
        while let Some(host) = pending.first().cloned() {
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
            pending.remove(0);
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
                    // The node that starts the process resolves the engine
                    // package its launch names at its own Build Server.
                    executable: String::new(),
                    arguments: Vec::new(),
                    inherit,
                    // The node that starts the process writes its launch file.
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
/// test made). An engine of another version of the engine interface than this daemon's, or one
/// that states another number, is refused in the welcome with the typed refusal.
///
/// # Errors
///
/// When the welcome cannot be written, or the engine is refused (the refusal, as the error).
pub(crate) fn welcome(
    writer: &mut Connection,
    hello: &wire::Hello,
    engine: u64,
) -> anyhow::Result<()> {
    let refusal = wire::Refusal::of_version(hello.major).or_else(|| {
        (hello.engine != engine).then_some(wire::Refusal {
            reason: Some(wire::refusal::Reason::Engine(wire::EngineMismatch {
                stated: hello.engine,
                expected: engine,
            })),
        })
    });
    writer.deliver(&wire::Welcome {
        build: proto::BUILD.to_owned(),
        refusal,
        event_window: crate::engine::client::EVENT_WINDOW,
    })?;
    match refusal {
        Some(refusal) => Err(anyhow::Error::new(refusal).context("refused the engine")),
        None => Ok(()),
    }
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
    Ok(Some((info.into(), ended)))
}

/// Reads the reply `id` on a control connection. The daemon's service answers every request on
/// it, so the read ends with the reply or with the connection's end.
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
                        started[index] = Some((info.into(), state));
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
        let prepared =
            prepared.map_err(|_| anyhow::anyhow!("the preparation at {host} panicked"))??;
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

/// What one node's Storage Server keeps for the deployment, as `GET /admin/config` reports it
/// (`quotas`): the node's host identity, its quota of every GPU the deployment uses, what the host
/// admits of new host storage, and what the deployment's tasks copied there by priority.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct NodeUsage {
    pub(crate) host: String,
    pub(crate) quotas: Vec<storage::protocol::Quota>,
    pub(crate) host_storage: storage::protocol::HostStorage,
    pub(crate) tasks: Vec<storage::protocol::TaskTotal>,
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
    let socket = ensure_build_server(settings)
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
/// operator at its selected revision.
///
/// This is the one selection of an apply. The daemon asks `cover` for each interface the
/// deployment uses: the engine, the scheduling policy and every kernel operator of this build.
/// A package the deployment names takes the revision it states, else its latest installed one.
/// Every candidate of an operator is recorded as well, one slot per package with the package as
/// its part: for each descriptor key the engine later asks `cover` to select among exactly
/// these revisions, so a package or a revision installed after this call is run by nobody until
/// a configuration is applied again, and a later revision of one of them is a package change a
/// configuration comparison sees. Every other host is then asked for exactly these revisions
/// ([`prepare_epoch`]).
///
/// # Errors
///
/// Fails when the Build Server cannot be reached, or holds no package, or not the revision, the
/// deployment names for an interface.
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
    fix_revision(&mut client, proto::module::ENGINE, &mut complete.engine)?;
    fix_revision(&mut client, proto::module::POLICY, &mut complete.policy)?;
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
            // engine selects among them for its own devices.
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

/// Runs `prepare` at one host's Build Server, which this process reaches at its socket, for the
/// module set `filled`, whose revisions are fixed, and returns it with the parameters each
/// package runs with: its manifest's defaults under the deployment's arguments.
fn prepare_at(
    server: &BuildServer,
    filled: &crate::epoch::ModuleSet,
) -> anyhow::Result<crate::epoch::ModuleSet> {
    let policy_slot = proto::module::ModuleSlot::all(proto::module::POLICY);
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
    let world_size = world_size(&args.engine_config)?;
    place_on(args, engine, interfaces, &hosts[0], world_size)
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

    /// A worker group whose ranks spread over two hosts has one engine per host, the engine
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

    /// The launches of the deployment file `file` of the repository's `configs/`, whose hosts are
    /// `rank-0`, the engine host, and `rank-1`: one engine per host, numbered 1 and 2, with each
    /// launch's host, first rank, devices, world size, whether it is the scheduler node, and epoch.
    fn two_host_launches(file: &str) -> Vec<(String, u32, Vec<u32>, u32, bool, u64)> {
        let configs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
        let layout = std::fs::read_to_string(configs.join(file)).expect("reading the layout");
        let (stated, _) =
            crate::options::resolve_text(&layout, &[]).expect("the layout is a configuration");
        let options = ServerArgs {
            model: Some("/models/checkpoint".to_owned()),
            host_identity: "rank-0".to_owned(),
            worker_groups: stated.worker_groups,
            ..Default::default()
        };
        let args = EngineArgs::new(
            options.engine_config().expect("the deployment file places"),
            modules(),
        );
        let interfaces = Interfaces {
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
        // One text replica and one flow replica on each host's four devices: the text replicas
        // are ranks 0-3 (`rank-0`) and 4-7 (`rank-1`), the flow replicas ranks 8-11 and 12-15.
        assert_eq!(
            two_host_launches("minimax_h3/fasth3-ulysses4x2-two-host.toml"),
            vec![
                ("rank-0".to_owned(), 0, vec![0, 1, 2, 3], 16, true, 1),
                ("rank-1".to_owned(), 4, vec![0, 1, 2, 3], 16, false, 1),
            ]
        );

        // A tensor-parallel attention group on `rank-0` (ranks 0 and 1) and its expert group on
        // `rank-1` (ranks 2 and 3).
        assert_eq!(
            two_host_launches("afd/qwen3_moe.toml"),
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

        let world = control.target_world(&args, &[9]).unwrap();
        let (launch, inherit) = &world[0];
        assert_eq!(world.len(), 1);
        assert!(inherit.is_empty(), "a world of one host reserves nothing");
        assert_eq!(launch.engine, 9);
        assert_eq!(launch.epoch, 3);
        assert_eq!(launch.processes, "unix:/run/processes.sock");
        assert_eq!(launch.requests, control.request_address());
        assert_eq!(launch.queue_depth, 4);
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
            data_parallel_size: std::num::NonZeroUsize::new(2).unwrap(),
            ..Default::default()
        };
        options.device_config.devices = vec![0, 1];
        let config = options.engine_config().expect("the configuration resolves");

        // The deployment's first epoch is numbered as its first engine is.
        let epoch = control.next_engine();
        let launches = crate::leader::epoch_launches(&config, &modules(), epoch).unwrap();
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
