//! The daemon's service thread: the process interface it serves, on the
//! leader to engines and other nodes' daemons, on every other node to the
//! leader.
//!
//! The leader replies to an engine's request for its own node itself and
//! forwards a request for another node to that node's daemon, tagged with
//! the engine connection it serves; the reply and the node's exits come
//! back the same way. A member daemon serves the leader's forwarded requests
//! until the leader's connection closes, and then ends its processes.
//!
//! One thread does all of it, waiting on every connection and a wake pipe
//! at once and polling its processes for exits at a short interval, so no
//! request waits behind another node's.
//!
//! A hot restart of the daemon stops the thread between two rounds and
//! hands everything it serves to the successor daemon: its listeners and
//! connections as descriptors, with the bytes each received and has not
//! taken, its pending stops, its processes and its claims
//! ([`ServiceHandoff`]). The successor's thread goes on from there; no peer
//! sees a connection close.

use std::collections::HashMap;
use std::net::IpAddr;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use prost::Message as _;
use proto::process::{
    self, Downstream, Event, Exit, Exited, Forwarded, FromMember, Hello, Interface, Node,
    NodesKnown, Replied, Reply, Request, Spawned, Stopped, ToMember, TransfersThrottled, Upstream,
    UsageReport, Welcome, downstream, event, from_member, hello, reply, request, to_member,
    upstream,
};
use service::{Connection, Handed, Listener};

use crate::supervision::claims::{self, ClaimHolder};
use crate::supervision::processes::Greeting;
use crate::supervision::supervisor::{ProcessesHandoff, Supervisor};

/// How often the node's processes are polled for exits while nothing else happens.
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// What the service thread reads its stop from.
pub(crate) struct Stop {
    pub(crate) requested: Arc<AtomicBool>,
    /// Set to stop for a hot restart: the thread hands what it serves over
    /// instead of ending its processes.
    pub(crate) handoff: Arc<AtomicBool>,
    pub(crate) wake: UnixStream,
}

impl Stop {
    fn requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    fn handoff(&self) -> bool {
        self.handoff.load(Ordering::Acquire)
    }

    /// Consumes the wake bytes the stop wrote.
    fn drain(&self) {
        use std::io::Read as _;

        let mut bytes = [0_u8; 64];
        while let Ok(read) = (&self.wake).read(&mut bytes) {
            if read == 0 {
                break;
            }
        }
    }
}

/// The local part of the process interface: the supervisor and the claims
/// of this node.
pub(crate) struct Local {
    pub(crate) supervisor: Supervisor,
    pub(crate) claims: ClaimHolder,
    /// This node's address other nodes reach; `None` on a single node.
    pub(crate) address: Option<IpAddr>,
    /// Reads this host's idle devices (`Request::Devices`).
    pub(crate) devices: super::DeviceProbe,
    /// Where each preparation this node runs off its service sends its reply, with the connection
    /// that asked (`Local::prepare`).
    preparations: Sender<(u64, Reply)>,
    /// The replies of the preparations that ended, which the service delivers (`Local::prepared`).
    prepared: Receiver<(u64, Reply)>,
}

impl Local {
    /// This node's local part: its supervisor and claims, its address other nodes reach (`None`
    /// on a single node) and its device probe.
    pub(crate) fn new(
        supervisor: Supervisor,
        claims: ClaimHolder,
        address: Option<IpAddr>,
        devices: super::DeviceProbe,
    ) -> Self {
        let (preparations, prepared) = std::sync::mpsc::channel();
        Self {
            supervisor,
            claims,
            address,
            devices,
            preparations,
            prepared,
        }
    }

    /// Hands the supervisor's processes and the claims over: their state,
    /// and their descriptors appended to `descriptors`.
    fn hand_over(&mut self, descriptors: &mut Vec<OwnedFd>) -> anyhow::Result<LocalHandoff> {
        let (claims, claim_descriptors) = self.claims.handoff()?;
        let (processes, process_descriptors) = self.supervisor.hand_over()?;
        let handoff = LocalHandoff {
            claims,
            claim_descriptors: claim_descriptors.len(),
            processes,
            process_descriptors: process_descriptors.len(),
            address: self.address,
        };
        descriptors.extend(claim_descriptors);
        descriptors.extend(process_descriptors);
        Ok(handoff)
    }

    /// Goes on with the handoff of [`Local::hand_over`], taking its
    /// descriptors from `descriptors`.
    fn adopt(
        host: String,
        devices: super::DeviceProbe,
        handoff: LocalHandoff,
        descriptors: &mut std::vec::IntoIter<OwnedFd>,
    ) -> anyhow::Result<Self> {
        let claims: Vec<OwnedFd> = descriptors
            .by_ref()
            .take(handoff.claim_descriptors)
            .collect();
        let processes: Vec<OwnedFd> = descriptors
            .by_ref()
            .take(handoff.process_descriptors)
            .collect();
        anyhow::ensure!(
            claims.len() == handoff.claim_descriptors
                && processes.len() == handoff.process_descriptors,
            "the service's handoff lacks descriptors"
        );
        Ok(Self::new(
            Supervisor::adopt(host, handoff.processes, processes)?,
            ClaimHolder::adopt(handoff.address, handoff.claims, claims)?,
            handoff.address,
            devices,
        ))
    }

    /// Replies to one request for this node on behalf of `engine`. `reporter` is
    /// the connection that asked, which a launch's exit report goes down.
    ///
    /// `None` for a preparation, which is answered once it ends: its reply then
    /// comes from [`Local::prepared`].
    fn reply(&mut self, engine: u64, reporter: u64, request: &Request) -> Option<Reply> {
        let mut reply = Reply {
            id: request.id,
            ..Reply::default()
        };
        match &request.kind {
            Some(request::Kind::PrepareModules(prepare)) => {
                self.prepare(reporter, request.id, prepare.clone());
                return None;
            }
            Some(request::Kind::Spawn(spawn)) => {
                match self.supervisor.spawn(engine, reporter, spawn) {
                    Ok(pid) => reply.kind = Some(reply::Kind::Spawned(Spawned { pid })),
                    Err(error) => reply.error = error,
                }
            }
            Some(request::Kind::Stop(stop)) => {
                self.supervisor.stop(engine, &stop.group, stop.generation);
                reply.kind = Some(reply::Kind::Stopped(Stopped {}));
            }
            Some(request::Kind::Reserve(reserve)) => {
                // A reservation is filed under the engine that will spawn the
                // process inheriting it, which the request states: the asking
                // connection may be the deployment's own client, and it may
                // be re-established between the reservation and the launch
                // without changing whose socket it is.
                let owner = if reserve.engine == 0 {
                    engine
                } else {
                    reserve.engine
                };
                match self.supervisor.reserve(owner, reserve, self.address) {
                    Ok(reserved) => reply.kind = Some(reply::Kind::Reserved(reserved)),
                    Err(error) => reply.error = error,
                }
            }
            Some(request::Kind::Claim(claim)) => match self.claims.claim(claim) {
                Ok(claimed) => reply.kind = Some(reply::Kind::Claimed(claimed)),
                Err(error) => reply.error = format!("{error:#}"),
            },
            Some(request::Kind::PublishEpochs(publish)) => {
                match self
                    .claims
                    .publish_epochs(&publish.deployment, &publish.epochs)
                {
                    Ok(()) => {
                        reply.kind = Some(reply::Kind::EpochsPublished(
                            proto::process::EpochsPublished {},
                        ))
                    }
                    Err(error) => reply.error = format!("{error:#}"),
                }
            }
            Some(request::Kind::KeepWeights(keep)) => {
                match self.claims.keep_weights(
                    &keep.deployment,
                    &keep.epochs,
                    keep.to_host,
                    &keep.cache_keys,
                ) {
                    Ok(kept) => reply.kind = Some(reply::Kind::WeightsKept(kept)),
                    Err(error) => reply.error = format!("{error:#}"),
                }
            }
            Some(request::Kind::Usage(query)) => match self.claims.usage(&query.deployment) {
                Ok(usage) => reply.kind = Some(reply::Kind::Usage(UsageReport { usage })),
                Err(error) => reply.error = format!("{error:#}"),
            },
            Some(request::Kind::ThrottleTransfers(throttle)) => {
                match self.claims.throttle(&throttle.deployment, throttle.halve) {
                    Ok(()) => {
                        reply.kind = Some(reply::Kind::TransfersThrottled(TransfersThrottled {}));
                    }
                    Err(error) => reply.error = format!("{error:#}"),
                }
            }
            Some(request::Kind::Nodes(_)) => {
                reply.error = "the leader replies for the deployment's nodes".to_owned();
            }
            Some(request::Kind::Devices(_)) => match (self.devices)() {
                Ok(available) => {
                    reply.kind = Some(reply::Kind::Devices(process::DeviceInventory { available }))
                }
                Err(error) => reply.error = error,
            },
            None => reply.error = "an empty request".to_owned(),
        }
        Some(reply)
    }

    /// Prepares the module set `prepare` names at this node's Build Server on a thread of its
    /// own, which sends the reply to request `id` of the connection `reporter` to
    /// [`Local::prepared`] once the Build Server answered: a store that misses the engine's
    /// executable builds it for minutes, while the service goes on serving the node's other
    /// requests and reporting its processes' exits.
    ///
    /// A preparation in flight when the service hands over to a successor sends its reply to no
    /// one: the leader's wait for it ends with the deadline of the preparation as a whole.
    fn prepare(&self, reporter: u64, id: u64, prepare: process::PrepareModules) {
        let settings = self.supervisor.build_server().cloned();
        let finished = self.preparations.clone();
        let started = std::thread::Builder::new()
            .name("prepare".into())
            .spawn(move || {
                let mut reply = Reply {
                    id,
                    ..Reply::default()
                };
                match crate::engine::control::prepare_node(settings.as_ref(), &prepare) {
                    Ok(prepared) => reply.kind = Some(reply::Kind::ModulesPrepared(prepared)),
                    Err(error) => reply.error = error,
                }
                // A service that ended reads no reply: its asker sees the connection close.
                let _ = finished.send((reporter, reply));
            });
        if let Err(error) = started {
            let _ = self.preparations.send((
                reporter,
                Reply {
                    id,
                    error: format!("starting this node's preparation: {error}"),
                    kind: None,
                },
            ));
        }
    }

    /// The replies of the preparations that ended since the last call, each with the connection
    /// that asked.
    fn prepared(&self) -> Vec<(u64, Reply)> {
        self.prepared.try_iter().collect()
    }
}

/// A member daemon as the leader holds it.
pub(crate) struct Membership {
    pub(crate) node_rank: u32,
    pub(crate) host: String,
    pub(crate) connection: Connection,
    /// The requests forwarded to the member that it has not replied to, by the connection that
    /// asked and the request's id: a member that is lost replies to none of them, so the leader
    /// fails each with the node's loss. A successor daemon starts with none, and a request
    /// forwarded before a hot restart ends at its asker's own deadline when its node is lost: no
    /// preparation runs across a restart, which waits for the update in flight to end.
    pub(crate) forwarded: Vec<(u64, u64)>,
}

/// The leader's address members join at, and the connections that have not
/// said hello yet.
pub(crate) struct Joins {
    listener: Listener,
    connecting: Vec<Connection>,
}

impl Joins {
    pub(crate) fn new(listener: Listener) -> Self {
        Self {
            listener,
            connecting: Vec::new(),
        }
    }

    /// Accepts the connections waiting and replies to every hello that
    /// arrived: a member of a node rank other than the leader's and every
    /// member's in `members` joins, any other hello is refused. Returns the
    /// members that joined.
    ///
    /// # Errors
    ///
    /// Fails when the listener cannot accept.
    pub(crate) fn admit(&mut self, members: &[Membership]) -> anyhow::Result<Vec<Membership>> {
        while let Some(connection) = self.listener.try_accept()? {
            self.connecting.push(connection);
        }
        let mut joined = Vec::new();
        let mut index = 0;
        while index < self.connecting.len() {
            let hello = match self.connecting[index].try_recv::<Hello>() {
                Ok(Some(hello)) => hello,
                Ok(None) => {
                    index += 1;
                    continue;
                }
                Err(_) => {
                    self.connecting.swap_remove(index);
                    continue;
                }
            };
            let mut connection = self.connecting.swap_remove(index);
            let member = match &hello.peer {
                Some(hello::Peer::Join(join)) => Some(join.clone()),
                _ => None,
            };
            let refusal =
                refusal(&hello, |peer| matches!(peer, hello::Peer::Join(_))).or_else(|| {
                    let member = member.as_ref()?;
                    (member.node_rank == 0
                        || members
                            .iter()
                            .chain(&joined)
                            .any(|known| known.node_rank == member.node_rank))
                    .then(|| format!("node rank {} is taken", member.node_rank))
                });
            // A member joins in its own version, which the connection speaks for its life.
            let welcome = Welcome {
                major: if refusal.is_none() {
                    hello.major
                } else {
                    proto::ENGINE_INTERFACE_VERSION
                },
                build: proto::BUILD.to_owned(),
                refusal: refusal.clone().unwrap_or_default(),
                nodes: Vec::new(),
            };
            let sent = connection.deliver(&welcome).is_ok();
            if let (None, Some(member), true) = (refusal, member, sent) {
                tracing::info!(node_rank = member.node_rank, host = %member.host, "a node joined");
                joined.push(Membership {
                    node_rank: member.node_rank,
                    host: member.host,
                    connection,
                    forwarded: Vec::new(),
                });
            }
        }
        Ok(joined)
    }

    /// The descriptors a wait for joins polls.
    fn descriptors(&self) -> impl Iterator<Item = std::os::fd::RawFd> + '_ {
        std::iter::once(self.listener.as_raw_fd())
            .chain(self.connecting.iter().map(Connection::as_raw_fd))
    }
}

/// The deployment's node of `member`, as engines learn it.
fn member_node(member: &Membership) -> Node {
    Node {
        node_rank: member.node_rank,
        host: member.host.clone(),
        address: member
            .connection
            .peer_addr()
            .map(|address| address.ip().to_string())
            .unwrap_or_default(),
    }
}

/// An engine connection to the leader.
struct EngineConnection {
    /// The engine whose process link this is, once it stated one: the number that link holds in
    /// `process_links` until the connection closes.
    process_of: Option<u64>,
    /// The engine whose request link this is, once it stated one, held the same way.
    requests_of: Option<u64>,
    connection: Connection,
    /// Whether its hello was welcomed.
    welcomed: bool,
    /// The owner this connection's requests are served for: the engine
    /// number a process link states, or `EPOCH_ENGINE` for a peer that is no
    /// process of the deployment — a worker's client, the daemon's own client.
    ///
    /// It is set by the hello and never by arrival: a reservation one
    /// connection makes is inherited by a spawn another connection sends, so
    /// the two must be served for the same owner even though the daemon
    /// accepted them under different arrival numbers.
    owner: u64,
}

/// The owner id of the engines the updater starts: it is no
/// connection, so no disconnect ever releases them, and the service keeps
/// them for the daemon's life.
pub(crate) const EPOCH_ENGINE: u64 = 0;

/// The first key among the clients of the socket. A connection travels as one
/// number to a member and back with a launch's reply and exit report, so a
/// client's key must not be a number an engine holds or may state.
const FIRST_CLIENT: u64 = 1 << 63;

/// The connection `id` names, in the registry its range says.
fn connection_of<'a>(
    id: u64,
    engines: &'a mut HashMap<u64, EngineConnection>,
    clients: &'a mut HashMap<u64, EngineConnection>,
) -> Option<&'a mut EngineConnection> {
    if id >= FIRST_CLIENT {
        clients.get_mut(&id)
    } else {
        engines.get_mut(&id)
    }
}

/// Reports one exit: down the connection that asked for the launch, and to the
/// updater, whose switch reads every exit of the deployment.
fn report_exit(
    reporter: u64,
    exit: Exit,
    exits: &Sender<Exit>,
    wake: &Sender<()>,
    engines: &mut HashMap<u64, EngineConnection>,
    clients: &mut HashMap<u64, EngineConnection>,
) {
    let _ = exits.send(exit.clone());
    let _ = wake.send(());
    if let Some(peer) = connection_of(reporter, engines, clients) {
        let _ = peer.connection.deliver(&downstream_exit(exit));
    }
}

/// Runs the leader's service until `stop`: engines connect on `listener`,
/// `members` joined before, and other nodes may join at `joins`. Returns
/// what it served when `stop` asks for a hot restart's handoff, and `None`
/// once it stopped.
/// The daemon's process table: the processes this node supervises, which
/// `GET /admin/config` reports and which a successor daemon keeps.
pub(crate) type ProcessTable = std::sync::Arc<std::sync::Mutex<Vec<crate::epoch::ManagedProcess>>>;

/// Publishes the node's live processes into `table`, keeping what the
/// updater recorded of each of them — its epoch, its build and the state it
/// reported — and dropping the rows of processes that ended.
fn publish(local: &Local, table: &ProcessTable) {
    let live = local.supervisor.live();
    let mut rows = ::service::lock(table);
    rows.retain(|row| live.iter().any(|process| same_process(row, process)));
    for process in live {
        if let Some(row) = rows.iter_mut().find(|row| same_process(row, &process)) {
            row.pid = Some(process.pid);
            continue;
        }
        rows.push(crate::epoch::ManagedProcess {
            group: process.group,
            rank: u64::from(process.rank),
            pid: Some(process.pid),
            // The epoch an engine serves is its own report's, which the
            // updater records against this row.
            epoch: None,
            build: proto::BUILD.to_owned(),
            generation: process.generation,
            // The daemon sees it running; whether it serves requests, parks
            // or failed is what the process itself reports, which the
            // updater records against this row.
            state: crate::epoch::ProcessState::Serving,
        });
    }
}

/// Whether `row` and `process` are the same process: one launch generation of
/// one worker group's rank.
fn same_process(
    row: &crate::epoch::ManagedProcess,
    process: &crate::supervision::supervisor::ProcessHandoff,
) -> bool {
    row.group == process.group
        && row.rank == u64::from(process.rank)
        && row.generation == process.generation
}

#[expect(
    clippy::too_many_arguments,
    reason = "the service takes each channel end and resource it owns separately: they come from \
              different owners, and `resume` takes the same ends from a handoff"
)]
pub(crate) fn serve_leader(
    host: &str,
    local: Local,
    listener: Listener,
    joins: Option<Joins>,
    members: Vec<Membership>,
    commands: Receiver<(u64, process::Command)>,
    reports: Sender<(u64, process::Report)>,
    requests: Receiver<(Request, Sender<Reply>)>,
    greetings: Sender<(u64, Greeting)>,
    exits: Sender<Exit>,
    wake: Sender<()>,
    processes: ProcessTable,
    stop: &Stop,
) -> anyhow::Result<Option<(ServiceHandoff, Vec<OwnedFd>)>> {
    let mut nodes = vec![Node {
        node_rank: 0,
        host: host.to_owned(),
        address: local
            .address
            .map(|address| address.to_string())
            .unwrap_or_default(),
    }];
    nodes.extend(members.iter().map(member_node));
    nodes.sort_by_key(|node| node.node_rank);
    run_leader(
        LeaderService {
            host: host.to_owned(),
            local,
            listener,
            joins,
            members,
            nodes,
            engines: HashMap::new(),
            next_engine: 1,
            clients: HashMap::new(),
            next_client: FIRST_CLIENT,
            process_links: HashMap::new(),
            request_links: HashMap::new(),
            stops: HashMap::new(),
            commands,
            reports,
            requests,
            greetings,
            exits,
            pending: HashMap::new(),
            wake,
            processes,
        },
        stop,
    )
}

/// Everything the leader's service holds between two rounds.
pub(crate) struct LeaderService {
    host: String,
    local: Local,
    listener: Listener,
    /// Where other nodes join the running deployment, when it names the
    /// leader's address.
    joins: Option<Joins>,
    members: Vec<Membership>,
    nodes: Vec<Node>,
    engines: HashMap<u64, EngineConnection>,
    next_engine: u64,
    /// Peers of the process socket that are not processes of the deployment -
    /// the daemon's own client, and the clients an engine's workers open.
    /// They are served and never numbered, so no command can be addressed to
    /// one of them.
    clients: HashMap<u64, EngineConnection>,
    next_client: u64,
    /// The connection each engine's process link is on, by engine number: the
    /// one link that receives that engine's commands, and the reason a second
    /// process link for the same number is refused.
    process_links: HashMap<u64, u64>,
    /// The connection each engine's request link is on, by engine number. It
    /// never receives a command, and a second request link is refused.
    request_links: HashMap<u64, u64>,
    /// Stops of an engine, by its request id, awaiting the members'
    /// replies.
    stops: HashMap<(u64, u64), PendingStop>,
    /// The commands the daemon sends down to engines
    /// (`supervision::ProcessCommands`), each carried to its engine's
    /// connection.
    commands: Receiver<(u64, process::Command)>,
    /// Where each `Upstream::Report` an engine sends goes,
    /// with the engine that sent it.
    reports: Sender<(u64, process::Report)>,
    /// The updater's requests, each with where its reply goes: a
    /// request for this node gets its reply at once, one for another node
    /// when that node's daemon's reply arrives.
    requests: Receiver<(Request, Sender<Reply>)>,
    /// Every peer that greeted the leader, with its build and its kind: the
    /// updater binds a peer to the launch it started once the peer's
    /// report names it.
    greetings: Sender<(u64, Greeting)>,
    /// Every supervised process that exited, for the updater's switch.
    exits: Sender<Exit>,
    /// The updater's requests a node has not replied to yet, by request id.
    pending: HashMap<u64, Sender<Reply>>,
    /// A token per thing the updater has to read: a greeting, a report or
    /// an exit wakes the thread that advances the switch.
    wake: Sender<()>,
    /// The daemon's process table, which this service keeps current as
    /// processes start and end.
    processes: ProcessTable,
}

/// Runs the leader's service from `service` until `stop`.
pub(crate) fn run_leader(
    service: LeaderService,
    stop: &Stop,
) -> anyhow::Result<Option<(ServiceHandoff, Vec<OwnedFd>)>> {
    let LeaderService {
        host,
        mut local,
        listener,
        mut joins,
        mut members,
        mut nodes,
        mut engines,
        mut next_engine,
        mut clients,
        mut next_client,
        mut process_links,
        mut request_links,
        mut stops,
        commands,
        reports,
        requests,
        greetings,
        exits,
        mut pending,
        wake,
        processes,
    } = service;
    let host = host.as_str();
    loop {
        // What this node supervises now, for the daemon's process table: the
        // loop turns on events, so a reader of `GET /admin/config` sees the
        // processes that started and ended.
        publish(&local, &processes);
        if stop.handoff() {
            let service = LeaderService {
                host: host.to_owned(),
                local,
                listener,
                joins,
                members,
                nodes,
                engines,
                next_engine,
                clients,
                next_client,
                process_links,
                request_links,
                stops,
                commands,
                reports,
                requests,
                greetings,
                exits,
                pending,
                wake,
                processes,
            };
            return service.hand_over().map(Some);
        }
        if stop.requested() {
            local.supervisor.terminate_all();
            return Ok(None);
        }
        while let Some(connection) = listener.try_accept()? {
            // The arrival numbering skips the numbers engines state:
            // an engine a launch named owns its number for the deployment's
            // life, and a later arrival must not be given it.
            while engines.contains_key(&next_engine) {
                next_engine += 1;
            }
            engines.insert(
                next_engine,
                EngineConnection {
                    connection,
                    welcomed: false,
                    process_of: None,
                    requests_of: None,
                    owner: next_engine,
                },
            );
            next_engine += 1;
        }

        // Nodes joining the running deployment: an engine that connects
        // from now on learns them, so the next epoch may run ranks there.
        if let Some(joins) = &mut joins {
            for member in joins.admit(&members)? {
                nodes.push(member_node(&member));
                nodes.sort_by_key(|node| node.node_rank);
                members.push(member);
            }
        }

        // The daemon's commands down to an engine: each
        // goes to its engine's connection, and a connection that cannot be
        // written closes that engine below.
        let mut closed = Vec::new();
        while let Ok((engine, command)) = commands.try_recv() {
            let Some(peer) = engines.get_mut(&engine) else {
                tracing::warn!(engine, "a command names an engine that is not connected");
                continue;
            };
            if peer
                .connection
                .deliver(&Downstream {
                    kind: Some(downstream::Kind::Event(process::Event {
                        kind: Some(event::Kind::Command(command)),
                    })),
                })
                .is_err()
            {
                closed.push((engine, peer.owner));
            }
        }

        // The updater's requests: a spawn, stop or claim it asks
        // for is executed here, by this node's supervisor and claims or by a
        // member's daemon, and the reply goes back the way it came.
        while let Ok((request, waiter)) = requests.try_recv() {
            match route(
                EPOCH_ENGINE,
                EPOCH_ENGINE,
                &request,
                host,
                &mut local,
                &mut members,
                &nodes,
                &mut stops,
            ) {
                Some(reply) => {
                    let _ = waiter.send(reply);
                }
                None => {
                    pending.insert(request.id, waiter);
                }
            }
        }

        // The engines that stated an engine number this round, by the
        // number their arrival gave them: applied once their hellos are read.
        let mut renumbered: Vec<(u64, u64)> = Vec::new();
        // Peers that stated no engine number and hold none: the daemon's own
        // client of this socket, and the clients an engine's workers open.
        // Nothing addresses them, so they are kept out of the numbered registry
        // and served for what they ask.
        let mut unkeyed: Vec<u64> = Vec::new();
        // The request links that greeted this round, by the number they state:
        // filed beside the clients, because a command never goes down one.
        let mut request_links_new: Vec<(u64, u64)> = Vec::new();
        // Whether a peer greeted this round and is filed under its key at its
        // end: what it sent behind its hello is read at once, in the next
        // round, rather than after a wait its socket would not end.
        let mut refiled = false;
        // Engines: their hellos and requests; a closed engine takes its
        // processes on every node with it.
        for (&engine, peer) in &mut engines {
            loop {
                if !peer.welcomed {
                    match peer.connection.try_recv::<Hello>() {
                        Ok(Some(hello)) => {
                            // An engine says which of its links this is and
                            // which engine number it holds: its process link
                            // takes the epoch's commands, its request link
                            // carries the epoch's requests, and a client of the
                            // socket is neither.
                            let (stated, link) = match &hello.peer {
                                Some(hello::Peer::Engine(engine)) => (
                                    Some(engine.engine).filter(|number| *number != 0),
                                    engine.interface(),
                                ),
                                _ => (None, Interface::Unspecified),
                            };
                            let mut refusal = refusal(&hello, |peer| {
                                matches!(peer, hello::Peer::Engine(_) | hello::Peer::Frontend(_))
                            });
                            if refusal.is_none() {
                                refusal = match (link, stated) {
                                    // A link that receives commands must say
                                    // which engine's commands it receives.
                                    (Interface::Process, None) => Some(
                                        "a process link states no engine number: no command could                                          be addressed to it"
                                            .to_owned(),
                                    ),
                                    // A process has one command link. A second
                                    // one is a deployment fault, and the first
                                    // keeps the commands it is serving.
                                    (Interface::Process, Some(number)) => {
                                        process_links.get(&number).map(|held| {
                                            format!(
                                                "engine {number} already has its process link on \
                                                 connection {held}; this connection states the \
                                                 same engine"
                                            )
                                        })
                                    }
                                    // A process has one request link.
                                    (Interface::Requests, Some(number)) => {
                                        request_links.get(&number).map(|held| {
                                            format!(
                                                "engine {number} already has its request link on \
                                                 connection {held}; this connection states the \
                                                 same engine"
                                            )
                                        })
                                    }
                                    _ => None,
                                };
                            }
                            // A peer is welcomed in its own version, which
                            // the connection speaks for its life: an engine
                            // of the build this daemon replaced keeps serving.
                            let welcome = Welcome {
                                major: if refusal.is_none() {
                                    hello.major
                                } else {
                                    proto::ENGINE_INTERFACE_VERSION
                                },
                                build: proto::BUILD.to_owned(),
                                refusal: refusal.clone().unwrap_or_default(),
                                nodes: nodes.clone(),
                            };
                            if peer.connection.deliver(&welcome).is_err() || refusal.is_some() {
                                closed.push((engine, peer.owner));
                                break;
                            }
                            let frontend = matches!(hello.peer, Some(hello::Peer::Frontend(_)));
                            // What this connection's requests are served
                            // for: the number it states when it is a process
                            // of the deployment, and the epoch's own owner
                            // when it is a client. A frontend states no
                            // number and is commanded by the updater, so
                            // it keeps the one its arrival gave it.
                            peer.owner = match (frontend, link, stated) {
                                (false, Interface::Process | Interface::Requests, Some(number)) => {
                                    number
                                }
                                (true, _, _) => engine,
                                _ => EPOCH_ENGINE,
                            };
                            tracing::info!(
                                engine,
                                stated = stated.unwrap_or(0),
                                interface = link.as_str_name(),
                                build = %hello.build,
                                frontend,
                                "a process connected"
                            );
                            // Three kinds of peer reach this point: a process's
                            // command link, which the epoch addresses by number
                            // and which is filed under it; a process's request
                            // link, which never receives a command and so is
                            // kept out of the numbered registry; and a client of
                            // the socket, which is served for its queries and
                            // holds no number at all.
                            if frontend {
                                // The frontend is commanded by the updater
                                // like an engine, and is filed by whatever
                                // number it arrived with.
                            } else {
                                match (link, stated) {
                                    // Only a process link takes a number: it is
                                    // the connection a command goes down.
                                    (Interface::Process, Some(number)) => {
                                        // Reserved as the link is read, so that a
                                        // second process link for the same
                                        // engine in this same round is refused
                                        // rather than displacing the first.
                                        process_links.insert(number, engine);
                                        renumbered.push((engine, number));
                                    }
                                    (Interface::Requests, Some(number)) => {
                                        request_links_new.push((engine, number));
                                    }
                                    // Everything else - a worker's client, the
                                    // daemon's own client, a link that states a
                                    // number but not the interface that would
                                    // make it a command target - is served and
                                    // never numbered.
                                    _ => unkeyed.push(engine),
                                }
                            }
                            // The updater binds this connection to the
                            // launch it started when the peer's first report
                            // names the launch generation. A client
                            // sends no report, and takes no greeting.
                            if frontend || link == Interface::Process {
                                let _ = greetings.send((
                                    peer.owner,
                                    Greeting {
                                        build: hello.build.clone(),
                                        frontend,
                                    },
                                ));
                            }
                            let _ = wake.send(());
                            peer.welcomed = true;
                            // Every peer but the frontend is filed under
                            // another key at the end of this round, and a
                            // request is served for the key it is read under:
                            // the launches it asks for and their exits name
                            // that key. A request that followed the hello
                            // closely is therefore read next round, under the
                            // key the peer keeps.
                            if frontend {
                                continue;
                            }
                            refiled = true;
                            break;
                        }
                        Ok(None) => break,
                        Err(_) => {
                            closed.push((engine, peer.owner));
                            break;
                        }
                    }
                }
                match peer.connection.try_recv::<Upstream>() {
                    Ok(Some(Upstream {
                        kind: Some(upstream::Kind::Request(request)),
                    })) => {
                        if let Some(reply) = route(
                            engine,
                            peer.owner,
                            &request,
                            host,
                            &mut local,
                            &mut members,
                            &nodes,
                            &mut stops,
                        ) && peer.connection.deliver(&downstream_reply(reply)).is_err()
                        {
                            closed.push((engine, peer.owner));
                            break;
                        }
                    }
                    Ok(Some(Upstream {
                        kind: Some(upstream::Kind::Report(report)),
                    })) => {
                        // Reports go to the daemon's switch; a
                        // daemon that stopped reading them drops them.
                        let _ = reports.send((peer.owner, report));
                        let _ = wake.send(());
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(_) => {
                        closed.push((engine, peer.owner));
                        break;
                    }
                }
            }
        }
        // Clients of the process socket: the daemon's own, and the ones an
        // engine's workers open. They ask (a device inventory, a storage
        // quota, a launch) and get a response; they are never commanded, and a
        // client that goes takes with it the launches it asked for and nothing
        // else: the deployment's owner stays, whose processes the updater
        // named.
        let mut gone = Vec::new();
        for (&client, peer) in &mut clients {
            loop {
                match peer.connection.try_recv::<Upstream>() {
                    Ok(Some(Upstream {
                        kind: Some(upstream::Kind::Request(request)),
                    })) => {
                        if let Some(reply) = route(
                            client,
                            EPOCH_ENGINE,
                            &request,
                            host,
                            &mut local,
                            &mut members,
                            &nodes,
                            &mut stops,
                        ) && peer.connection.deliver(&downstream_reply(reply)).is_err()
                        {
                            gone.push(client);
                            break;
                        }
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(_) => {
                        gone.push(client);
                        break;
                    }
                }
            }
        }
        for client in gone {
            if let Some(number) = clients.remove(&client).and_then(|peer| peer.requests_of) {
                request_links.remove(&number);
            }
            stops.retain(|&(asker, _), _| asker != client);
            local.supervisor.release(client);
            for member in &mut members {
                let _ = member.connection.deliver(&ToMember {
                    kind: Some(to_member::Kind::Released(client)),
                });
            }
        }

        for (key, engine) in closed {
            // A link ends with its connection: a process relaunched under the number of one that
            // died states that number again, and a link kept here would refuse it.
            if let Some(number) = engines.remove(&key).and_then(|peer| peer.process_of) {
                process_links.remove(&number);
            }
            if engine == EPOCH_ENGINE {
                // A client of the socket owns no process: it goes, and takes
                // nothing with it — not the updater's processes, which
                // are named by the same owner.
                tracing::info!(client = key, "a client of the process socket disconnected");
                continue;
            }
            stops.retain(|&(owner, _), _| owner != engine);
            tracing::info!(engine, "an engine disconnected; ending its processes");
            local.supervisor.release(engine);
            for member in &mut members {
                let _ = member.connection.deliver(&ToMember {
                    kind: Some(to_member::Kind::Released(engine)),
                });
            }
        }

        // Members: replies and exits on their nodes, for their engines.
        let mut lost = Vec::new();
        for member in &mut members {
            loop {
                match member.connection.try_recv::<FromMember>() {
                    Ok(Some(message)) => {
                        match message.kind {
                            Some(from_member::Kind::Replied(Replied {
                                engine,
                                reply: Some(reply),
                            })) => {
                                member
                                    .forwarded
                                    .retain(|&forwarded| forwarded != (engine, reply.id));
                                match stops.get_mut(&(engine, reply.id)) {
                                    // A stop gets its reply once every node stopped
                                    // the group.
                                    Some(stop) => {
                                        if !reply.error.is_empty() {
                                            stop.reply.error = reply.error;
                                        }
                                        stop.awaiting.retain(|&rank| rank != member.node_rank);
                                        if !stop.awaiting.is_empty() {
                                            continue;
                                        }
                                        let Some(stop) = stops.remove(&(engine, reply.id)) else {
                                            continue;
                                        };
                                        deliver(
                                            engine,
                                            stop.reply,
                                            &mut pending,
                                            &mut engines,
                                            &mut clients,
                                        );
                                        continue;
                                    }
                                    None => {
                                        deliver(
                                            engine,
                                            reply,
                                            &mut pending,
                                            &mut engines,
                                            &mut clients,
                                        );
                                        continue;
                                    }
                                }
                            }
                            Some(from_member::Kind::Exited(Exited {
                                engine,
                                exit: Some(exit),
                            })) => {
                                report_exit(
                                    engine,
                                    exit,
                                    &exits,
                                    &wake,
                                    &mut engines,
                                    &mut clients,
                                );
                                continue;
                            }
                            _ => continue,
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        // A lost node's processes died with its daemon; its
                        // engines notice their workers' connections closing.
                        tracing::warn!(node_rank = member.node_rank, host = %member.host, %error, "the daemon of a node is lost");
                        lost.push(member.node_rank);
                        break;
                    }
                }
            }
        }

        // A lost node runs nothing more: requests for it fail from here on, the
        // requests it had not replied to fail now, naming it, and the stops that
        // awaited it get their reply without it.
        if !lost.is_empty() {
            for member in members
                .iter()
                .filter(|member| lost.contains(&member.node_rank))
            {
                for &(asker, id) in &member.forwarded {
                    let failure = Reply {
                        id,
                        error: format!(
                            "the daemon of node {} ({}) was lost before it replied",
                            member.node_rank, member.host
                        ),
                        kind: None,
                    };
                    deliver(asker, failure, &mut pending, &mut engines, &mut clients);
                }
            }
            members.retain(|member| !lost.contains(&member.node_rank));
            nodes.retain(|node| !lost.contains(&node.node_rank));
            let mut settled = Vec::new();
            for (&key, stop) in &mut stops {
                stop.awaiting.retain(|rank| !lost.contains(rank));
                if stop.awaiting.is_empty() {
                    settled.push(key);
                }
            }
            for key in settled {
                if let Some(stop) = stops.remove(&key) {
                    deliver(key.0, stop.reply, &mut pending, &mut engines, &mut clients);
                }
            }
        }

        // The clients leave the numbered registry before any command is read:
        // a command names a launch's engine, and a client that happened to
        // arrive first must not be the connection it names.
        for (arrived, number) in request_links_new.drain(..) {
            if let Some(mut peer) = engines.remove(&arrived) {
                let client = next_client;
                next_client += 1;
                request_links.insert(number, arrived);
                peer.requests_of = Some(number);
                tracing::info!(
                    client,
                    engine = number,
                    "a request link is not a command target"
                );
                clients.insert(client, peer);
            }
        }
        for arrived in unkeyed.drain(..) {
            if let Some(peer) = engines.remove(&arrived) {
                let client = next_client;
                next_client += 1;
                tracing::info!(
                    client,
                    engine = arrived,
                    "a client of the process socket is not a process of the deployment"
                );
                clients.insert(client, peer);
            }
        }

        // The numbers engines stated take effect before this round's
        // commands go down, so a command the updater sends right after a
        // process reported itself ready already reaches that process.
        let filed = file_stated(&mut engines, renumbered.drain(..), &mut next_engine);
        for (fresh, stated) in filed.displaced {
            tracing::info!(
                engine = fresh,
                stated,
                "a peer was moved off the number an engine states"
            );
        }
        for (number, arrived) in filed.numbered {
            if let Some(connection) = engines.get_mut(&number) {
                connection.process_of = Some(number);
            }
            process_links.insert(number, arrived);
            tracing::info!(
                engine = number,
                arrived,
                "an engine is numbered by the number it states"
            );
        }

        // This node's exits.
        for (reporter, exit) in local.supervisor.exits() {
            report_exit(reporter, exit, &exits, &wake, &mut engines, &mut clients);
        }

        // This node's preparations that ended.
        for (reporter, reply) in local.prepared() {
            deliver(reporter, reply, &mut pending, &mut engines, &mut clients);
        }

        let mut descriptors = vec![listener.as_raw_fd(), stop.wake.as_raw_fd()];
        descriptors.extend(joins.iter().flat_map(Joins::descriptors));
        descriptors.extend(engines.values().map(|peer| peer.connection.as_raw_fd()));
        descriptors.extend(clients.values().map(|peer| peer.connection.as_raw_fd()));
        descriptors.extend(members.iter().map(|member| member.connection.as_raw_fd()));
        if !refiled {
            wait(&descriptors, EXIT_POLL_INTERVAL);
        }
        stop.drain();
    }
}

/// What [`file_stated`] did: each engine filed under the number it stated, with the number its
/// arrival gave it, and each peer moved off a stated number, with its new number.
struct Filed {
    /// `(stated, arrived)` per engine filed under its stated number.
    numbered: Vec<(u64, u64)>,
    /// `(new number, stated)` per peer that held a stated number without stating it.
    displaced: Vec<(u64, u64)>,
}

/// Files every engine that stated its number this round under that number.
///
/// `moves` holds `(arrived, stated)`: the number an engine's arrival gave it and the number it
/// stated. Every moving engine is taken out by its arrival number before any is filed, so a
/// stated number that is another moving engine's arrival number (two engines that arrived in
/// each other's order) is never confused with it, and a command for a stated number reaches the
/// engine that stated it. A peer that holds a stated number without having stated it (a worker's
/// client, or a client of this interface, which nothing addresses) moves to the next number no
/// peer holds. A move whose arrival number holds no peer (it closed) is skipped.
fn file_stated<C>(
    engines: &mut HashMap<u64, C>,
    moves: impl IntoIterator<Item = (u64, u64)>,
    next_engine: &mut u64,
) -> Filed {
    let moving: Vec<(u64, u64, C)> = moves
        .into_iter()
        .filter_map(|(arrived, stated)| {
            engines.remove(&arrived).map(|peer| (arrived, stated, peer))
        })
        .collect();
    let mut filed = Filed {
        numbered: Vec::with_capacity(moving.len()),
        displaced: Vec::new(),
    };
    let stated: Vec<u64> = moving.iter().map(|(_, stated, _)| *stated).collect();
    for (arrived, number, peer) in moving {
        if let Some(displaced) = engines.remove(&number) {
            while engines.contains_key(next_engine) || stated.contains(next_engine) {
                *next_engine += 1;
            }
            engines.insert(*next_engine, displaced);
            filed.displaced.push((*next_engine, number));
            *next_engine += 1;
        }
        engines.insert(number, peer);
        filed.numbered.push((number, arrived));
    }
    filed
}

/// A stop the leader applied on its own node, awaiting the members' replies.
struct PendingStop {
    /// This node's reply, carrying the last error a member reports.
    reply: Reply,
    /// The node ranks of the members that have not replied yet.
    awaiting: Vec<u32>,
}

/// Delivers one reply: to the updater when the engines' owner
/// asked, otherwise down the connection that asked.
fn deliver(
    engine: u64,
    reply: Reply,
    pending: &mut HashMap<u64, Sender<Reply>>,
    engines: &mut HashMap<u64, EngineConnection>,
    clients: &mut HashMap<u64, EngineConnection>,
) {
    if engine == EPOCH_ENGINE {
        if let Some(waiter) = pending.remove(&reply.id) {
            let _ = waiter.send(reply);
        }
    } else if let Some(peer) = connection_of(engine, engines, clients) {
        let _ = peer.connection.deliver(&downstream_reply(reply));
    }
}

/// Routes one engine request: served here for this node, forwarded to the
/// node's daemon otherwise, whose reply comes back later. A stop applies to
/// every node and gets its reply once every node stopped the group: it waits in
/// `stops` for the members' replies.
///
/// `engine` is the connection the reply goes back down, and `owner` the
/// process the request is served *for* — the engine number a process states,
/// or the epoch's own owner for a client. They differ for a client, which holds
/// no engine number but whose launches are served for the deployment.
#[expect(
    clippy::too_many_arguments,
    reason = "the request, its two engine numbers and the service's routing state it reads and \
              updates are distinct borrows of the service loop's own locals"
)]
fn route(
    engine: u64,
    owner: u64,
    request: &Request,
    host: &str,
    local: &mut Local,
    members: &mut [Membership],
    nodes: &[Node],
    stops: &mut HashMap<(u64, u64), PendingStop>,
) -> Option<Reply> {
    if let Some(request::Kind::Nodes(_)) = &request.kind {
        return Some(Reply {
            id: request.id,
            kind: Some(reply::Kind::Nodes(NodesKnown {
                nodes: nodes.to_vec(),
            })),
            ..Reply::default()
        });
    }
    // A claim that states no Storage Server settings takes the deployment's, here or on the
    // member it goes to: the server a node starts for the claim then serves with them.
    let stated;
    let request = match &request.kind {
        Some(request::Kind::Claim(claim)) if claim.storage.is_none() => {
            let mut claim = claim.clone();
            claim.storage = Some(local.claims.settings().into());
            stated = Request {
                id: request.id,
                kind: Some(request::Kind::Claim(claim)),
            };
            &stated
        }
        _ => request,
    };
    let target = match &request.kind {
        Some(request::Kind::Spawn(spawn)) => Some(spawn.host.as_str()),
        Some(request::Kind::Reserve(reserve)) => Some(reserve.host.as_str()),
        Some(request::Kind::Claim(claim)) => Some(claim.host.as_str()),
        Some(request::Kind::Usage(query)) => Some(query.host.as_str()),
        Some(request::Kind::KeepWeights(keep)) => Some(keep.host.as_str()),
        Some(request::Kind::ThrottleTransfers(throttle)) => Some(throttle.host.as_str()),
        Some(request::Kind::Devices(query)) => Some(query.host.as_str()),
        Some(request::Kind::PublishEpochs(publish)) => Some(publish.host.as_str()),
        Some(request::Kind::PrepareModules(prepare)) => Some(prepare.host.as_str()),
        Some(request::Kind::Stop(_) | request::Kind::Nodes(_)) | None => None,
    };
    let forward = |member: &mut Membership| {
        member
            .connection
            .deliver(&ToMember {
                kind: Some(to_member::Kind::Forwarded(Forwarded {
                    engine,
                    owner,
                    request: Some(request.clone()),
                })),
            })
            .is_ok()
    };
    match target {
        None => {
            // A member whose connection fails is lost: its processes died
            // with it, and the receive that follows removes it.
            let awaiting: Vec<u32> = members
                .iter_mut()
                .filter_map(|member| forward(member).then_some(member.node_rank))
                .collect();
            // Only a stop and an empty request name no node, and both are answered at once: the
            // one request answered later, a preparation, names its node.
            let reply = local.reply(owner, engine, request)?;
            if awaiting.is_empty() {
                return Some(reply);
            }
            stops.insert((engine, request.id), PendingStop { reply, awaiting });
            None
        }
        Some(target) if target == host => local.reply(owner, engine, request),
        Some(target) => match members.iter_mut().find(|member| member.host == target) {
            Some(member) => {
                if forward(member) {
                    member.forwarded.push((engine, request.id));
                    None
                } else {
                    Some(failed(
                        request,
                        format!("the daemon of node {} is lost", member.node_rank),
                    ))
                }
            }
            None => Some(failed(
                request,
                format!("no daemon of the deployment runs on host {target}"),
            )),
        },
    }
}

/// Runs a member's service until the leader's connection closes or `stop`.
/// Returns what it served when `stop` asks for a hot restart's handoff, and
/// `None` once it stopped.
pub(crate) fn serve_member(
    mut local: Local,
    mut leader: Connection,
    stop: &Stop,
) -> anyhow::Result<Option<(ServiceHandoff, Vec<OwnedFd>)>> {
    loop {
        if stop.handoff() {
            let mut descriptors = Vec::new();
            let local = local.hand_over(&mut descriptors)?;
            let leader = leader.hand_over()?;
            let handoff = ServiceHandoff::Member {
                local,
                leader: ConnectionHandoff::of(&leader),
            };
            let mut ordered = vec![leader.descriptor];
            ordered.extend(descriptors);
            return Ok(Some((handoff, ordered)));
        }
        if stop.requested() {
            local.supervisor.terminate_all();
            return Ok(None);
        }
        loop {
            match leader.try_recv::<ToMember>() {
                Ok(Some(ToMember {
                    kind:
                        Some(to_member::Kind::Forwarded(Forwarded {
                            engine,
                            owner,
                            request: Some(request),
                        })),
                })) => {
                    // A preparation is answered once it ends, below.
                    if let Some(reply) = local.reply(owner, engine, &request) {
                        leader.deliver(&FromMember {
                            kind: Some(from_member::Kind::Replied(Replied {
                                engine,
                                reply: Some(reply),
                            })),
                        })?;
                    }
                }
                Ok(Some(ToMember {
                    kind: Some(to_member::Kind::Released(engine)),
                })) => local.supervisor.release(engine),
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(error) => {
                    tracing::info!(%error, "the leader's connection closed; ending this node's processes");
                    local.supervisor.terminate_all();
                    return Ok(None);
                }
            }
        }
        for (engine, exit) in local.supervisor.exits() {
            leader.deliver(&FromMember {
                kind: Some(from_member::Kind::Exited(Exited {
                    engine,
                    exit: Some(exit),
                })),
            })?;
        }
        // The preparations that ended, whose replies go up as any other: the
        // round's wait is short enough that a reply waits at most that long.
        for (engine, reply) in local.prepared() {
            leader.deliver(&FromMember {
                kind: Some(from_member::Kind::Replied(Replied {
                    engine,
                    reply: Some(reply),
                })),
            })?;
        }
        wait(
            &[leader.as_raw_fd(), stop.wake.as_raw_fd()],
            EXIT_POLL_INTERVAL,
        );
        stop.drain();
    }
}

/// What a daemon's service hands its successor in a hot restart: its state,
/// beside the descriptors that travel with it in the order the fields name
/// them.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) enum ServiceHandoff {
    Leader {
        host: String,
        /// The process socket's endpoint; its descriptor comes first.
        listener: ::service::Endpoint,
        /// The members' endpoint, when the deployment names one; its descriptor follows the
        /// process socket's. A node whose hello is still awaited joins again.
        joins: Option<::service::Endpoint>,
        /// Each engine connection, by its identity, after the listener.
        engines: Vec<(u64, bool, ConnectionHandoff)>,
        next_engine: u64,
        /// Each client's connection, after the engines'.
        #[serde(default)]
        clients: Vec<ClientHandoff>,
        /// Each member's connection, after the clients'.
        members: Vec<(u32, String, ConnectionHandoff)>,
        /// The deployment's nodes (`proto::process::Node`, encoded).
        nodes: Vec<Vec<u8>>,
        /// Each pending stop: its engine and request, this node's reply
        /// (encoded) and the members that have not replied.
        stops: Vec<(u64, u64, Vec<u8>, Vec<u32>)>,
        /// The claims' and processes' descriptors follow the members'.
        local: LocalHandoff,
    },
    Member {
        /// The leader's connection, whose descriptor comes first.
        leader: ConnectionHandoff,
        local: LocalHandoff,
    },
}

/// A connection's state, its descriptor travelling apart.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct ConnectionHandoff {
    unix: bool,
    received: Vec<u8>,
}

impl ConnectionHandoff {
    fn of(connection: &Handed) -> Self {
        Self {
            unix: connection.unix,
            received: connection.received.clone(),
        }
    }

    fn resume(self, descriptor: OwnedFd) -> anyhow::Result<Connection> {
        Ok(Connection::resume(Handed {
            descriptor,
            unix: self.unix,
            received: self.received,
        })?)
    }
}

/// A client of the process socket, its descriptor travelling apart. A client
/// that reached `clients` greeted, so nothing of its hello is left to read.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct ClientHandoff {
    /// Its key among the clients.
    id: u64,
    /// The owner its requests are served for.
    owner: u64,
    /// The engine whose request link it is, when it is one.
    requests_of: Option<u64>,
    connection: ConnectionHandoff,
}

/// The handoff of the supervisor's processes and the node's claims.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct LocalHandoff {
    claims: claims::Handoff,
    claim_descriptors: usize,
    processes: ProcessesHandoff,
    process_descriptors: usize,
    address: Option<IpAddr>,
}

impl LeaderService {
    /// The handoff of everything the service holds.
    fn hand_over(self) -> anyhow::Result<(ServiceHandoff, Vec<OwnedFd>)> {
        let Self {
            host,
            mut local,
            listener,
            joins,
            members,
            nodes,
            engines,
            next_engine,
            // A client of the process socket goes on with the rest: a daemon's
            // control connection is one, and the connection its engine's process
            // lives under outlives a query.
            clients,
            // The successor numbers its clients on from those it adopts.
            next_client: _,
            // The link maps are derived from the connections that are handed
            // over: the successor's first look at a greeted peer files it.
            process_links: _,
            request_links: _,
            stops,
            // The command channel is the daemon's, not the service's: the
            // successor makes its own ends when it resumes (`ProcessCommands`).
            commands: _,
            reports: _,
            requests: _,
            greetings: _,
            exits: _,
            pending: _,
            wake: _,
            // The table is the daemon's, like the command channel: the
            // successor's own service publishes into it.
            processes: _,
        } = self;
        let (listener_descriptor, endpoint) = listener.hand_over();
        let mut descriptors = vec![listener_descriptor];
        let joins = joins.map(|joins| {
            let (descriptor, endpoint) = joins.listener.hand_over();
            descriptors.push(descriptor);
            endpoint
        });
        let mut engine_handoffs = Vec::with_capacity(engines.len());
        for (id, engine) in engines {
            let connection = engine.connection.hand_over()?;
            engine_handoffs.push((id, engine.welcomed, ConnectionHandoff::of(&connection)));
            descriptors.push(connection.descriptor);
        }
        let mut client_handoffs = Vec::with_capacity(clients.len());
        for (id, client) in clients {
            let connection = client.connection.hand_over()?;
            client_handoffs.push(ClientHandoff {
                id,
                owner: client.owner,
                requests_of: client.requests_of,
                connection: ConnectionHandoff::of(&connection),
            });
            descriptors.push(connection.descriptor);
        }
        let mut member_handoffs = Vec::with_capacity(members.len());
        for member in members {
            let connection = member.connection.hand_over()?;
            member_handoffs.push((
                member.node_rank,
                member.host,
                ConnectionHandoff::of(&connection),
            ));
            descriptors.push(connection.descriptor);
        }
        let local = local.hand_over(&mut descriptors)?;
        Ok((
            ServiceHandoff::Leader {
                host,
                listener: endpoint,
                joins,
                engines: engine_handoffs,
                next_engine,
                clients: client_handoffs,
                members: member_handoffs,
                nodes: nodes.iter().map(prost::Message::encode_to_vec).collect(),
                stops: stops
                    .into_iter()
                    .map(|((engine, request), pending)| {
                        (
                            engine,
                            request,
                            pending.reply.encode_to_vec(),
                            pending.awaiting,
                        )
                    })
                    .collect(),
                local,
            },
            descriptors,
        ))
    }
}

/// Goes on with the service a predecessor handed over, with its
/// descriptors in the handoff's order, until `stop`.
#[expect(
    clippy::too_many_arguments,
    reason = "the service takes each channel end and resource it owns separately, as \
              `serve_leader` does, beside the state its predecessor handed over"
)]
pub(crate) fn resume(
    host: String,
    devices: super::DeviceProbe,
    handoff: ServiceHandoff,
    descriptors: Vec<OwnedFd>,
    commands: Receiver<(u64, process::Command)>,
    reports: Sender<(u64, process::Report)>,
    requests: Receiver<(Request, Sender<Reply>)>,
    greetings: Sender<(u64, Greeting)>,
    exits: Sender<Exit>,
    wake: Sender<()>,
    processes: ProcessTable,
    stop: &Stop,
) -> anyhow::Result<Option<(ServiceHandoff, Vec<OwnedFd>)>> {
    let mut descriptors = descriptors.into_iter();
    let mut next = || {
        descriptors
            .next()
            .ok_or_else(|| anyhow::anyhow!("the service's handoff lacks a descriptor"))
    };
    match handoff {
        ServiceHandoff::Leader {
            host: leader_host,
            listener: endpoint,
            joins,
            engines,
            next_engine,
            clients,
            members,
            nodes,
            stops,
            local,
        } => {
            let listener = Listener::resume(next()?, &endpoint)?;
            let joins = joins
                .map(|endpoint| anyhow::Ok(Joins::new(Listener::resume(next()?, &endpoint)?)))
                .transpose()?;
            let mut adopted_engines = HashMap::new();
            for (id, welcomed, connection) in engines {
                adopted_engines.insert(
                    id,
                    EngineConnection {
                        connection: connection.resume(next()?)?,
                        welcomed,
                        // A process link of the handoff is filed under the number
                        // its engine stated, which is the identity its
                        // requests are served for; a peer still unfiled
                        // keeps the key it arrived under until it greets.
                        process_of: None,
                        requests_of: None,
                        owner: id,
                    },
                );
            }
            let mut adopted_clients = HashMap::new();
            let mut request_links = HashMap::new();
            for client in clients {
                if let Some(number) = client.requests_of {
                    request_links.insert(number, client.id);
                }
                adopted_clients.insert(
                    client.id,
                    EngineConnection {
                        connection: client.connection.resume(next()?)?,
                        welcomed: true,
                        process_of: None,
                        requests_of: client.requests_of,
                        owner: client.owner,
                    },
                );
            }
            let next_client = adopted_clients
                .keys()
                .max()
                .map_or(FIRST_CLIENT, |client| (client + 1).max(FIRST_CLIENT));
            let mut adopted_members = Vec::with_capacity(members.len());
            for (node_rank, member_host, connection) in members {
                adopted_members.push(Membership {
                    node_rank,
                    host: member_host,
                    connection: connection.resume(next()?)?,
                    forwarded: Vec::new(),
                });
            }
            let mut rest: std::vec::IntoIter<OwnedFd> = descriptors.collect::<Vec<_>>().into_iter();
            let local = Local::adopt(host, devices, local, &mut rest)?;
            let nodes = nodes
                .iter()
                .map(|node| Node::decode(node.as_slice()))
                .collect::<Result<Vec<_>, _>>()?;
            let stops = stops
                .into_iter()
                .map(|(engine, request, reply, awaiting)| {
                    Ok((
                        (engine, request),
                        PendingStop {
                            reply: Reply::decode(reply.as_slice())?,
                            awaiting,
                        },
                    ))
                })
                .collect::<anyhow::Result<HashMap<_, _>>>()?;
            tracing::info!(
                engines = adopted_engines.len(),
                clients = adopted_clients.len(),
                members = adopted_members.len(),
                "the service goes on from the previous daemon"
            );
            run_leader(
                LeaderService {
                    host: leader_host,
                    local,
                    listener,
                    joins,
                    members: adopted_members,
                    nodes,
                    engines: adopted_engines,
                    next_engine,
                    clients: adopted_clients,
                    next_client,
                    process_links: HashMap::new(),
                    request_links,
                    stops,
                    commands,
                    reports,
                    requests,
                    greetings,
                    exits,
                    pending: HashMap::new(),
                    wake,
                    processes,
                },
                stop,
            )
        }
        ServiceHandoff::Member { leader, local } => {
            let leader = leader.resume(next()?)?;
            let mut rest: std::vec::IntoIter<OwnedFd> = descriptors.collect::<Vec<_>>().into_iter();
            let local = Local::adopt(host, devices, local, &mut rest)?;
            tracing::info!("the service goes on from the previous daemon");
            serve_member(local, leader, stop)
        }
    }
}

/// Why a hello is refused, when it is: a major version this daemon's build
/// does not speak ([`proto::speaks`]), or a peer of the wrong kind for the
/// socket it reached.
pub(crate) fn refusal(hello: &Hello, accepts: impl Fn(&hello::Peer) -> bool) -> Option<String> {
    if !proto::speaks(hello.major) {
        return Some(format!(
            "the peer speaks major version {} and this daemon {} to {}",
            hello.major,
            proto::ENGINE_INTERFACE_OLDEST,
            proto::ENGINE_INTERFACE_VERSION
        ));
    }
    match &hello.peer {
        Some(peer) if accepts(peer) => None,
        _ => Some("the peer is not one this socket serves".to_owned()),
    }
}

fn failed(request: &Request, error: String) -> Reply {
    Reply {
        id: request.id,
        error,
        kind: None,
    }
}

fn downstream_reply(reply: Reply) -> Downstream {
    Downstream {
        kind: Some(downstream::Kind::Reply(reply)),
    }
}

fn downstream_exit(exit: proto::process::Exit) -> Downstream {
    Downstream {
        kind: Some(downstream::Kind::Event(Event {
            kind: Some(event::Kind::Exit(exit)),
        })),
    }
}

/// Waits up to `timeout` for any of `descriptors` to become readable.
fn wait(descriptors: &[RawFd], timeout: Duration) {
    let mut polled: Vec<libc::pollfd> = descriptors
        .iter()
        .map(|&fd| libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        })
        .collect();
    // SAFETY: `polled` holds one valid pollfd per descriptor for the call.
    unsafe {
        libc::poll(
            polled.as_mut_ptr(),
            polled.len() as libc::nfds_t,
            timeout.as_millis() as libc::c_int,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two engines whose stated numbers are each other's arrival numbers (the second replica
    /// of an epoch connected first) are each filed under the number it stated, so a command
    /// for either reaches it, and a client holding a stated number moves to a free one.
    #[test]
    fn engines_that_arrived_in_each_others_order_are_filed_under_their_own_numbers() {
        let mut engines: HashMap<u64, &str> =
            HashMap::from([(11, "states 12"), (12, "states 11"), (13, "client")]);
        let mut next_engine = 13;
        let filed = file_stated(
            &mut engines,
            [(12, 11), (11, 12), (14, 13)],
            &mut next_engine,
        );

        assert_eq!(engines.get(&11), Some(&"states 11"));
        assert_eq!(engines.get(&12), Some(&"states 12"));
        assert_eq!(filed.numbered, vec![(11, 12), (12, 11)]);
        // The move from arrival 14 names no peer; the client keeps its number.
        assert_eq!(engines.get(&13), Some(&"client"));
        assert!(filed.displaced.is_empty());
    }

    #[test]
    fn a_client_on_a_stated_number_moves_to_a_number_no_peer_holds() {
        let mut engines: HashMap<u64, &str> =
            HashMap::from([(1, "client"), (2, "states 1"), (3, "other")]);
        let mut next_engine = 2;
        let filed = file_stated(&mut engines, [(2, 1)], &mut next_engine);

        assert_eq!(engines.get(&1), Some(&"states 1"));
        assert_eq!(engines.get(&3), Some(&"other"));
        let (fresh, stated) = filed.displaced[0];
        assert_eq!(stated, 1);
        assert_eq!(engines.get(&fresh), Some(&"client"));
        assert_eq!(engines.len(), 3);
    }
}
