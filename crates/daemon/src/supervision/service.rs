//! The daemon's service thread: the process interface it serves, on the
//! leader to engines and other nodes' daemons, on every other node to the
//! leader.
//!
//! The leader replies to an engine's request for its own node itself and
//! forwards a request for another node to that node's daemon, tagged with
//! the connection that asked; the reply and the node's exits come back the
//! same way. A member daemon serves the leader's forwarded requests until the
//! leader's connection closes, and then ends its processes.
//!
//! One thread does all of it. It blocks until something it can act on
//! happens: a connection's bytes or close, a listener's pending connection,
//! a supervised process's exit (each process's pidfd turns readable), or the
//! service's waker, which every other thread fires once it handed the thread
//! work (a command, a request, an ended preparation, a stop). No request
//! waits behind another node's, and nothing is observed by a clock.
//!
//! Every connection of the leader's process socket is known by the identity
//! the service gives it when it accepts it, which never changes; what the
//! connection is follows from its hello alone ([`Role`]): an engine's process
//! link, which commands reach by the engine number it states, an engine's
//! request link, or a client of the socket.
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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};

use anyhow::Context as _;
use prost::Message as _;
use proto::process::refusal::Reason;
use proto::process::{
    self, Downstream, Event, Exit, Exited, Forwarded, FromMember, Hello, Interface, Node,
    NodesKnown, ProgramConfigured, ProgramRecords, Refusal, Released, Replied, Reply, Request,
    Spawned, Stopped, StorageConfigured, ToMember, TransfersThrottled, Unnumbered, Unserved,
    Upstream, UsageReport, Welcome, downstream, event, from_member, hello, reply, request,
    to_member, upstream,
};
use service::{Connection, Handed, Listener, Waker};

use crate::supervision::claims::{self, ClaimHolder};
use crate::supervision::supervisor::{Asker, Owner, ProcessesHandoff, Supervisor};

/// What the service thread reads its stop from, and the waker every other thread fires once it
/// handed the thread work.
pub(crate) struct Stop {
    pub(crate) requested: Arc<AtomicBool>,
    /// Set to stop for a hot restart: the thread hands what it serves over
    /// instead of ending its processes.
    pub(crate) handoff: Arc<AtomicBool>,
    /// Fired by a stop, a command or request the daemon sends, and a preparation that ended.
    pub(crate) waker: Waker,
}

impl Stop {
    fn requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }

    fn handoff(&self) -> bool {
        self.handoff.load(Ordering::Acquire)
    }
}

/// The local part of the process interface: the supervisor and the claims
/// of this node.
pub(crate) struct Local {
    pub(crate) supervisor: Supervisor,
    pub(crate) claims: ClaimHolder,
    /// This node's address other nodes reach; `None` on a single node.
    pub(crate) address: Option<IpAddr>,
    /// Where each preparation this node runs off its service sends its reply, with its asker
    /// (`Local::prepare`), and the wake that tells the service thread one ended.
    preparations: super::ToService<(Asker, Reply)>,
    /// The replies of the preparations that ended, which the service delivers (`Local::prepared`).
    prepared: Receiver<(Asker, Reply)>,
}

/// How a KV program's instance started, as the storage protocol names it
/// ([`storage::protocol::ProgramStart`]): what [`process::ProgramConfigured`] carries.
pub(crate) fn start_name(start: storage::protocol::ProgramStart) -> &'static str {
    match start {
        storage::protocol::ProgramStart::Kept => "kept",
        storage::protocol::ProgramStart::Restored => "restored",
        storage::protocol::ProgramStart::Created => "created",
    }
}

impl Local {
    /// This node's local part: its supervisor and claims, its address other nodes reach (`None`
    /// on a single node); `wake` wakes the service thread once a preparation
    /// ended.
    pub(crate) fn new(
        supervisor: Supervisor,
        claims: ClaimHolder,
        address: Option<IpAddr>,
        wake: service::Wake,
    ) -> Self {
        let (preparations, prepared) = std::sync::mpsc::channel();
        Self {
            supervisor,
            claims,
            address,
            preparations: super::ToService::new(preparations, wake),
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
        handoff: LocalHandoff,
        descriptors: &mut std::vec::IntoIter<OwnedFd>,
        wake: service::Wake,
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
            wake,
        ))
    }

    /// Replies to one request for this node, served for `owner` as `asker` asked: a launch's exit
    /// report goes to its asker.
    ///
    /// `None` for a preparation, which is answered once it ends: its reply then
    /// comes from [`Local::prepared`].
    fn reply(&mut self, owner: Owner, asker: Asker, request: &Request) -> Option<Reply> {
        let mut reply = Reply {
            id: request.id,
            ..Reply::default()
        };
        match &request.kind {
            Some(request::Kind::PrepareModules(prepare)) => {
                self.prepare(asker, request.id, prepare.clone());
                return None;
            }
            Some(request::Kind::Spawn(spawn)) => match self.supervisor.spawn(owner, asker, spawn) {
                Ok(pid) => reply.kind = Some(reply::Kind::Spawned(Spawned { pid })),
                Err(error) => reply.error = error,
            },
            Some(request::Kind::Stop(stop)) => {
                self.supervisor
                    .stop(owner, asker, &stop.group, stop.generation);
                reply.kind = Some(reply::Kind::Stopped(Stopped {}));
            }
            Some(request::Kind::Reserve(reserve)) => {
                // A reservation is filed under the engine that will spawn the process inheriting
                // it, when the request states one: the asking connection may be the deployment's
                // own client, and it may be re-established between the reservation and the
                // launch without changing whose socket it is.
                let owner = reserve.engine.map_or(owner, Owner::Engine);
                match self.supervisor.reserve(owner, reserve, self.address) {
                    Ok(reserved) => reply.kind = Some(reply::Kind::Reserved(reserved)),
                    Err(error) => reply.error = error,
                }
            }
            Some(request::Kind::Claim(claim)) => match self.claims.claim(owner.claimant(), claim) {
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
            Some(request::Kind::Usage(query)) => match self
                .claims
                .usage(&query.deployment)
                .and_then(|usage| Ok(storage::protocol::encode(&usage)?))
            {
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
            Some(request::Kind::ConfigureStorage(configure)) => {
                // The service waits while the server takes the settings: a restart in place
                // hands the server over once nothing is in flight, and the engines' requests
                // reach the server meanwhile, on their own connections.
                let configured = configure
                    .storage
                    .as_ref()
                    .context("the request states no settings")
                    .and_then(|stated| {
                        storage::Settings::try_from(stated)
                            .map_err(anyhow::Error::msg)
                            .context("reading the Storage Server settings")
                    })
                    .and_then(|settings| self.claims.configure(&configure.deployment, settings));
                match configured {
                    Ok(()) => {
                        reply.kind = Some(reply::Kind::StorageConfigured(StorageConfigured {}));
                    }
                    Err(error) => reply.error = format!("{error:#}"),
                }
            }
            Some(request::Kind::ConfigureProgram(configure)) => {
                match self
                    .claims
                    .program(&configure.deployment, configure.program.clone())
                {
                    Ok(start) => {
                        reply.kind = Some(reply::Kind::ProgramConfigured(ProgramConfigured {
                            start: start.map_or("", start_name).to_owned(),
                        }));
                    }
                    Err(error) => reply.error = format!("{error:#}"),
                }
            }
            Some(request::Kind::ProgramRecords(query)) => match self
                .claims
                .program_records(&query.deployment)
                .and_then(|records| Ok(storage::protocol::encode(&records)?))
            {
                Ok(records) => {
                    reply.kind = Some(reply::Kind::ProgramRecords(ProgramRecords { records }));
                }
                Err(error) => reply.error = format!("{error:#}"),
            },
            Some(request::Kind::Nodes(_)) => {
                reply.error = "the leader replies for the deployment's nodes".to_owned();
            }
            None => reply.error = "an empty request".to_owned(),
        }
        Some(reply)
    }

    /// Prepares the module set `prepare` names at this node's Build Server on a thread of its
    /// own, which sends the reply to request `id` of `asker` to [`Local::prepared`] once the
    /// Build Server answered: a store that misses the engine's executable builds it for minutes,
    /// while the service goes on serving the node's other requests and reporting its processes'
    /// exits.
    ///
    /// A preparation in flight when the service hands over to a successor sends its reply to no
    /// one: the leader's wait for it ends with the connection that asked.
    fn prepare(&self, asker: Asker, id: u64, prepare: process::PrepareModules) {
        let settings = self.supervisor.build_server().cloned();
        let deployment = self.supervisor.deployment().to_owned();
        let runtime = self.supervisor.runtime().clone();
        let finished = self.preparations.clone();
        let started = std::thread::Builder::new()
            .name("prepare".into())
            .spawn(move || {
                let mut reply = Reply {
                    id,
                    ..Reply::default()
                };
                match crate::engine::control::prepare_node(
                    &runtime,
                    &deployment,
                    settings.as_ref(),
                    &prepare,
                ) {
                    Ok(prepared) => reply.kind = Some(reply::Kind::ModulesPrepared(prepared)),
                    Err(error) => reply.error = error,
                }
                // A service that ended reads no reply: its asker sees the connection close.
                let _ = finished.send((asker, reply));
            });
        if let Err(error) = started {
            let _ = self.preparations.send((
                asker,
                Reply {
                    id,
                    error: format!("starting this node's preparation: {error}"),
                    kind: None,
                },
            ));
        }
    }

    /// The replies of the preparations that ended since the last call, each with its asker.
    fn prepared(&self) -> Vec<(Asker, Reply)> {
        self.prepared.try_iter().collect()
    }

    /// The descriptors whose readiness this node's part acts on: one per supervised process,
    /// readable once it exited.
    fn descriptors(&self) -> impl Iterator<Item = RawFd> + '_ {
        self.supervisor.exit_descriptors()
    }
}

/// A member daemon as the leader holds it.
pub(crate) struct Membership {
    pub(crate) node_rank: u32,
    pub(crate) host: String,
    /// The address the member declared in its join (`proto::process::Join::address`).
    pub(crate) address: String,
    pub(crate) connection: Connection,
    /// The requests forwarded to the member that it has not replied to, by asker and request id:
    /// a member that is lost replies to none of them, so the leader fails each with the node's
    /// loss. A successor daemon starts with none, and a request forwarded before a hot restart
    /// ends with its asker's connection when its node is lost: no preparation runs across a
    /// restart, which waits for the update in flight to end.
    pub(crate) forwarded: Vec<(Asker, u64)>,
}

impl Membership {
    /// The deployment's node this member is, as engines learn it.
    fn node(&self) -> Node {
        Node {
            node_rank: self.node_rank,
            host: self.host.clone(),
            address: self.address.clone(),
        }
    }
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
                    .then(|| Refusal::because(Reason::RankTaken(member.node_rank)))
                });
            let welcome = Welcome {
                build: proto::BUILD.to_owned(),
                refusal,
                nodes: Vec::new(),
            };
            let sent = connection.deliver(&welcome).is_ok();
            if let (None, Some(member), true) = (refusal, member, sent) {
                tracing::info!(node_rank = member.node_rank, host = %member.host, address = %member.address, "a node joined");
                joined.push(Membership {
                    node_rank: member.node_rank,
                    host: member.host,
                    address: member.address,
                    connection,
                    forwarded: Vec::new(),
                });
            }
        }
        Ok(joined)
    }

    /// The descriptors a wait for joins polls: the listener, readable while a connection waits,
    /// and every connection whose hello has not arrived.
    pub(crate) fn poll_fds(&self) -> impl Iterator<Item = libc::pollfd> + '_ {
        std::iter::once(readable(self.listener.as_raw_fd())).chain(
            self.connecting
                .iter()
                .flat_map(|connection| connection.poll_fds()),
        )
    }
}

/// What a connection of the leader's process socket is, which its hello states and which never
/// changes afterwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum Role {
    /// Accepted; its hello has not arrived.
    Greeting,
    /// The process link of the engine of this number: the one connection commands for that
    /// engine go down, and the one its reports come up on.
    Process(u64),
    /// The request link of the engine of this number, which carries its requests and is never
    /// commanded.
    Requests(u64),
    /// A client of the socket that is no link of an engine: the daemon's own clients and the
    /// clients an engine's workers open. It asks and is answered, and nothing addresses it.
    Client,
}

impl Role {
    /// Whom this connection's requests are served for: the engine whose link it is, and the
    /// deployment's head for a client.
    fn owner(self) -> Owner {
        match self {
            Self::Process(engine) | Self::Requests(engine) => Owner::Engine(engine),
            Self::Greeting | Self::Client => Owner::Head,
        }
    }

    /// The role the hello `hello` states, or why it is refused: another version of the engine
    /// interface, a peer the process socket does not serve, a process link that states no engine
    /// number, or a link of an engine that holds that link already (`links`).
    fn of_hello(hello: &Hello, links: &Links) -> Result<Self, Refusal> {
        if let Some(refusal) = refusal(hello, |peer| matches!(peer, hello::Peer::Engine(_))) {
            return Err(refusal);
        }
        let (stated, interface) = match &hello.peer {
            Some(hello::Peer::Engine(engine)) => (
                Some(engine.engine).filter(|number| *number != 0),
                engine.interface(),
            ),
            _ => (None, Interface::Unspecified),
        };
        match (interface, stated) {
            // A link that receives commands must say which engine's commands it receives.
            (Interface::Process, None) => Err(Refusal::because(Reason::Unnumbered(Unnumbered {}))),
            // A process has one command link. A second one is a deployment fault, and the first
            // keeps the commands it is serving.
            (Interface::Process, Some(engine)) if links.process.contains_key(&engine) => {
                Err(Refusal::because(Reason::Linked(engine)))
            }
            (Interface::Process, Some(engine)) => Ok(Self::Process(engine)),
            // A process has one request link.
            (Interface::Requests, Some(engine)) if links.requests.contains_key(&engine) => {
                Err(Refusal::because(Reason::Linked(engine)))
            }
            (Interface::Requests, Some(engine)) => Ok(Self::Requests(engine)),
            // Everything else - a worker's client, the daemon's own client, a request link that
            // states no number - is served and never numbered.
            _ => Ok(Self::Client),
        }
    }
}

/// One connection of the leader's process socket.
struct Peer {
    connection: Connection,
    role: Role,
}

/// The connection each engine's links are on, by engine number.
#[derive(Default)]
struct Links {
    /// Each engine's process link: the connection its commands go down.
    process: HashMap<u64, u64>,
    /// Each engine's request link.
    requests: HashMap<u64, u64>,
}

impl Links {
    /// Records `role`'s link on connection `connection`.
    fn insert(&mut self, role: Role, connection: u64) {
        match role {
            Role::Process(engine) => {
                self.process.insert(engine, connection);
            }
            Role::Requests(engine) => {
                self.requests.insert(engine, connection);
            }
            Role::Greeting | Role::Client => {}
        }
    }

    /// Forgets `role`'s link: a link ends with its connection, so a process relaunched under the
    /// number of one that died states that number again.
    fn remove(&mut self, role: Role) {
        match role {
            Role::Process(engine) => {
                self.process.remove(&engine);
            }
            Role::Requests(engine) => {
                self.requests.remove(&engine);
            }
            Role::Greeting | Role::Client => {}
        }
    }
}

/// Reports one exit: to the asker of its launch, and to the updater, whose switch reads every
/// exit of the deployment.
fn report_exit(
    asker: Asker,
    exit: Exit,
    exits: &Sender<Exit>,
    wake: &Sender<()>,
    peers: &mut HashMap<u64, Peer>,
) {
    let _ = exits.send(exit.clone());
    let _ = wake.send(());
    if let Asker::Connection(connection) = asker
        && let Some(peer) = peers.get_mut(&connection)
    {
        let _ = peer.connection.deliver(&downstream_exit(exit));
    }
}

/// The daemon's process table: the processes this node supervises, which
/// `process` events report and which a successor daemon keeps.
pub(crate) type ProcessTable = std::sync::Arc<std::sync::Mutex<Vec<proto::admin::ManagedProcess>>>;

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
        rows.push(proto::admin::ManagedProcess {
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
            state: proto::admin::ProcessState::Serving,
        });
    }
}

/// Whether `row` and `process` are the same process: one launch generation of
/// one worker group's rank.
fn same_process(
    row: &proto::admin::ManagedProcess,
    process: &crate::supervision::supervisor::ProcessHandoff,
) -> bool {
    row.group == process.group
        && row.rank == u64::from(process.rank)
        && row.generation == process.generation
}

/// The updater's ends of the leader's service, which the service thread holds: the commands and
/// requests the updater sends, and where the service sends what it sees.
pub(crate) struct Updater {
    /// The commands the daemon sends down to engines (`supervision::ProcessCommands`), each
    /// carried to its engine's process link.
    pub(crate) commands: Receiver<(u64, process::Command)>,
    /// Where each `Upstream::Report` an engine sends on its process link goes, with the engine
    /// that sent it.
    pub(crate) reports: Sender<(u64, process::Report)>,
    /// The updater's requests, each with where its reply goes: a request for this node gets its
    /// reply at once, one for another node when that node's daemon's reply arrives.
    pub(crate) requests: Receiver<(Request, Sender<Reply>)>,
    /// Every engine whose process link greeted the leader, with the build its hello states: the
    /// updater checks the build of every engine it starts.
    pub(crate) greetings: Sender<(u64, String)>,
    /// Every supervised process that exited, for the updater's switch.
    pub(crate) exits: Sender<Exit>,
    /// A token per thing the updater has to read: a greeting, a report or an exit wakes the
    /// thread that advances the switch.
    pub(crate) wake: Sender<()>,
    /// The daemon's process table, which this service keeps current as processes start and end.
    pub(crate) processes: ProcessTable,
}

/// Runs the leader's service until `stop`: engines connect on `listener`, `members` joined
/// before, and other nodes may join at `joins`. Returns what it served when `stop` asks for a hot
/// restart's handoff, and `None` once it stopped.
pub(crate) fn serve_leader(
    host: &str,
    local: Local,
    listener: Listener,
    joins: Option<Joins>,
    members: Vec<Membership>,
    updater: Updater,
    stop: &mut Stop,
) -> anyhow::Result<Option<(ServiceHandoff, Vec<OwnedFd>)>> {
    let mut nodes = vec![Node {
        node_rank: 0,
        host: host.to_owned(),
        address: local
            .address
            .map(|address| address.to_string())
            .unwrap_or_default(),
    }];
    nodes.extend(members.iter().map(Membership::node));
    nodes.sort_by_key(|node| node.node_rank);
    run_leader(
        LeaderService {
            host: host.to_owned(),
            local,
            listener,
            joins,
            members,
            nodes,
            peers: HashMap::new(),
            next_connection: 1,
            links: Links::default(),
            stops: HashMap::new(),
            pending: HashMap::new(),
            updater,
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
    /// Every connection of the process socket, by the identity the service gave it when it
    /// accepted it.
    peers: HashMap<u64, Peer>,
    /// The identity the next accepted connection takes.
    next_connection: u64,
    /// The connection each engine's links are on.
    links: Links,
    /// Stops, by asker and request id, awaiting the members' replies.
    stops: HashMap<(Asker, u64), PendingStop>,
    /// The updater's requests a node has not replied to yet, by request id.
    pending: HashMap<u64, Sender<Reply>>,
    updater: Updater,
}

/// Runs the leader's service from `service` until `stop`.
pub(crate) fn run_leader(
    mut service: LeaderService,
    stop: &mut Stop,
) -> anyhow::Result<Option<(ServiceHandoff, Vec<OwnedFd>)>> {
    loop {
        // What this node supervises now, for the daemon's process table: the
        // loop turns on events, so a reader of the table sees the processes
        // that started and ended.
        publish(&service.local, &service.updater.processes);
        if stop.handoff() {
            return service.hand_over().map(Some);
        }
        if stop.requested() {
            service.local.supervisor.terminate_all();
            return Ok(None);
        }
        service.round()?;
        service.wait(stop)?;
    }
}

impl LeaderService {
    /// One round: takes everything that arrived and acts on it.
    fn round(&mut self) -> anyhow::Result<()> {
        while let Some(connection) = self.listener.try_accept()? {
            self.peers.insert(
                self.next_connection,
                Peer {
                    connection,
                    role: Role::Greeting,
                },
            );
            self.next_connection += 1;
        }

        // Nodes joining the running deployment: an engine that connects
        // from now on learns them, so the next epoch may run ranks there.
        if let Some(joins) = &mut self.joins {
            for member in joins.admit(&self.members)? {
                self.nodes.push(member.node());
                self.nodes.sort_by_key(|node| node.node_rank);
                self.members.push(member);
            }
        }

        // The daemon's commands down to an engine: each goes to its engine's process link, and
        // a connection that cannot be written closes below.
        let mut closed = Vec::new();
        while let Ok((engine, command)) = self.updater.commands.try_recv() {
            let Some(connection) = self.links.process.get(&engine).copied() else {
                tracing::warn!(engine, "a command names an engine that is not connected");
                continue;
            };
            let Some(peer) = self.peers.get_mut(&connection) else {
                continue;
            };
            let delivered = peer.connection.deliver(&Downstream {
                kind: Some(downstream::Kind::Event(process::Event {
                    kind: Some(event::Kind::Command(command)),
                })),
            });
            if delivered.is_err() {
                closed.push(connection);
            }
        }

        // The updater's requests: a spawn, stop or claim it asks for is executed here, by this
        // node's supervisor and claims or by a member's daemon, and the reply goes back the way
        // it came.
        while let Ok((request, waiter)) = self.updater.requests.try_recv() {
            match self.route(Owner::Head, Asker::Updater, &request) {
                Some(reply) => {
                    let _ = waiter.send(reply);
                }
                None => {
                    self.pending.insert(request.id, waiter);
                }
            }
        }

        // The connections of the process socket: their hellos, then their requests and reports,
        // in arrival order on each.
        let identities: Vec<u64> = self.peers.keys().copied().collect();
        for connection in identities {
            if !self.serve_peer(connection) {
                closed.push(connection);
            }
        }
        for connection in closed {
            self.close(connection);
        }

        self.serve_members();

        // This node's exits.
        for (asker, exit) in self.local.supervisor.exits() {
            report_exit(
                asker,
                exit,
                &self.updater.exits,
                &self.updater.wake,
                &mut self.peers,
            );
        }

        // This node's preparations that ended.
        for (asker, reply) in self.local.prepared() {
            self.deliver(asker, reply);
        }
        Ok(())
    }

    /// Reads everything connection `connection` sent: its hello, which states what it is, and
    /// then its requests and reports. Returns `false` once the connection is to close: it closed,
    /// failed, sent what does not decode, or was refused.
    fn serve_peer(&mut self, connection: u64) -> bool {
        loop {
            let Some(peer) = self.peers.get_mut(&connection) else {
                return true;
            };
            if peer.role == Role::Greeting {
                match peer.connection.try_recv::<Hello>() {
                    Ok(Some(hello)) => {
                        if !self.greet(connection, &hello) {
                            return false;
                        }
                        continue;
                    }
                    Ok(None) => return true,
                    Err(_) => return false,
                }
            }
            let role = peer.role;
            match peer.connection.try_recv::<Upstream>() {
                Ok(Some(Upstream {
                    kind: Some(upstream::Kind::Request(request)),
                })) => {
                    if let Some(reply) =
                        self.route(role.owner(), Asker::Connection(connection), &request)
                        && let Some(peer) = self.peers.get_mut(&connection)
                        && peer.connection.deliver(&downstream_reply(reply)).is_err()
                    {
                        return false;
                    }
                }
                Ok(Some(Upstream {
                    kind: Some(upstream::Kind::Report(report)),
                })) => match role {
                    // An engine reports on its process link; the updater reads every report.
                    Role::Process(engine) => {
                        let _ = self.updater.reports.send((engine, report));
                        let _ = self.updater.wake.send(());
                    }
                    _ => tracing::warn!(
                        connection,
                        ?role,
                        "a connection that is no process link sent a report"
                    ),
                },
                Ok(Some(_)) => {}
                Ok(None) => return true,
                Err(_) => return false,
            }
        }
    }

    /// Answers connection `connection`'s hello: it takes the role the hello states, or is
    /// refused. Returns whether it stays.
    fn greet(&mut self, connection: u64, hello: &Hello) -> bool {
        let role = Role::of_hello(hello, &self.links);
        let welcome = Welcome {
            build: proto::BUILD.to_owned(),
            refusal: role.err(),
            nodes: self.nodes.clone(),
        };
        let Some(peer) = self.peers.get_mut(&connection) else {
            return false;
        };
        let Ok(role) = role else {
            let _ = peer.connection.deliver(&welcome);
            return false;
        };
        if peer.connection.deliver(&welcome).is_err() {
            return false;
        }
        peer.role = role;
        self.links.insert(role, connection);
        tracing::info!(connection, ?role, build = %hello.build, "a process connected");
        if let Role::Process(engine) = role {
            let _ = self.updater.greetings.send((engine, hello.build.clone()));
            let _ = self.updater.wake.send(());
        }
        true
    }

    /// Closes connection `connection`: what it asked for ends with it, on every node, and, when
    /// it was an engine's process link, what that engine owns.
    fn close(&mut self, connection: u64) {
        let Some(peer) = self.peers.remove(&connection) else {
            return;
        };
        self.links.remove(peer.role);
        let engine = match peer.role {
            Role::Process(engine) => Some(engine),
            _ => None,
        };
        tracing::info!(connection, role = ?peer.role, "a connection of the process socket closed");
        self.stops
            .retain(|&(asker, _), _| asker != Asker::Connection(connection));
        self.local.supervisor.release(connection, engine);
        for member in &mut self.members {
            let _ = member.connection.deliver(&ToMember {
                kind: Some(to_member::Kind::Released(Released { connection, engine })),
            });
        }
    }

    /// Reads the members' replies and exits, and fails what a lost member owed.
    fn serve_members(&mut self) {
        let mut lost = Vec::new();
        let mut replies = Vec::new();
        let mut exits = Vec::new();
        for member in &mut self.members {
            loop {
                match member.connection.try_recv::<FromMember>() {
                    Ok(Some(FromMember {
                        kind:
                            Some(from_member::Kind::Replied(Replied {
                                connection,
                                reply: Some(reply),
                            })),
                    })) => {
                        let asker = Asker::of_wire(connection);
                        member
                            .forwarded
                            .retain(|&forwarded| forwarded != (asker, reply.id));
                        replies.push((member.node_rank, asker, reply));
                    }
                    Ok(Some(FromMember {
                        kind:
                            Some(from_member::Kind::Exited(Exited {
                                connection,
                                exit: Some(exit),
                            })),
                    })) => exits.push((Asker::of_wire(connection), exit)),
                    Ok(Some(_)) => {}
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
        for (node_rank, asker, reply) in replies {
            let key = (asker, reply.id);
            let Some(stop) = self.stops.get_mut(&key) else {
                self.deliver(asker, reply);
                continue;
            };
            // A stop gets its reply once every node stopped the group.
            if !reply.error.is_empty() {
                stop.reply.error = reply.error;
            }
            stop.awaiting.retain(|&rank| rank != node_rank);
            if stop.awaiting.is_empty()
                && let Some(stop) = self.stops.remove(&key)
            {
                self.deliver(asker, stop.reply);
            }
        }
        for (asker, exit) in exits {
            report_exit(
                asker,
                exit,
                &self.updater.exits,
                &self.updater.wake,
                &mut self.peers,
            );
        }

        // A lost node runs nothing more: requests for it fail from here on, the
        // requests it had not replied to fail now, naming it, and the stops that
        // awaited it get their reply without it.
        if lost.is_empty() {
            return;
        }
        let mut failures = Vec::new();
        for member in self
            .members
            .iter()
            .filter(|member| lost.contains(&member.node_rank))
        {
            for &(asker, id) in &member.forwarded {
                failures.push((
                    asker,
                    Reply {
                        id,
                        error: format!(
                            "the daemon of node {} ({}) was lost before it replied",
                            member.node_rank, member.host
                        ),
                        kind: None,
                    },
                ));
            }
        }
        for (asker, failure) in failures {
            self.deliver(asker, failure);
        }
        self.members
            .retain(|member| !lost.contains(&member.node_rank));
        self.nodes.retain(|node| !lost.contains(&node.node_rank));
        let mut settled = Vec::new();
        for (&key, stop) in &mut self.stops {
            stop.awaiting.retain(|rank| !lost.contains(rank));
            if stop.awaiting.is_empty() {
                settled.push(key);
            }
        }
        for key in settled {
            if let Some(stop) = self.stops.remove(&key) {
                self.deliver(key.0, stop.reply);
            }
        }
    }

    /// Blocks until the service has something to act on: a connection's or a member's bytes or
    /// close, a waiting connection or join, a supervised process's exit, or the waker. Frames a
    /// connection already buffered were taken this round, so the wait cannot sleep on work.
    fn wait(&self, stop: &mut Stop) -> anyhow::Result<()> {
        let mut descriptors = vec![readable(self.listener.as_raw_fd()), stop.waker.poll_fd()];
        if let Some(joins) = &self.joins {
            descriptors.extend(joins.poll_fds());
        }
        descriptors.extend(
            self.peers
                .values()
                .flat_map(|peer| peer.connection.poll_fds()),
        );
        descriptors.extend(
            self.members
                .iter()
                .flat_map(|member| member.connection.poll_fds()),
        );
        descriptors.extend(self.local.descriptors().map(readable));
        service::wait_ready(&mut descriptors, None)?;
        stop.waker.drain();
        Ok(())
    }

    /// Delivers one reply: to the updater when it asked, otherwise down the connection that
    /// asked.
    fn deliver(&mut self, asker: Asker, reply: Reply) {
        match asker {
            Asker::Updater => {
                if let Some(waiter) = self.pending.remove(&reply.id) {
                    let _ = waiter.send(reply);
                }
            }
            Asker::Connection(connection) => {
                if let Some(peer) = self.peers.get_mut(&connection) {
                    let _ = peer.connection.deliver(&downstream_reply(reply));
                }
            }
        }
    }

    /// Routes one request, served for `owner` as `asker` asked: served here for this node,
    /// forwarded to the node's daemon otherwise, whose reply comes back later. A stop applies to
    /// every node and gets its reply once every node stopped the group: it waits in `stops` for
    /// the members' replies.
    fn route(&mut self, owner: Owner, asker: Asker, request: &Request) -> Option<Reply> {
        if let Some(request::Kind::Nodes(_)) = &request.kind {
            return Some(Reply {
                id: request.id,
                kind: Some(reply::Kind::Nodes(NodesKnown {
                    nodes: self.nodes.clone(),
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
                claim.storage = Some(self.local.claims.settings().into());
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
            Some(request::Kind::PublishEpochs(publish)) => Some(publish.host.as_str()),
            Some(request::Kind::PrepareModules(prepare)) => Some(prepare.host.as_str()),
            Some(request::Kind::ConfigureStorage(configure)) => Some(configure.host.as_str()),
            Some(request::Kind::ConfigureProgram(configure)) => Some(configure.host.as_str()),
            Some(request::Kind::ProgramRecords(query)) => Some(query.host.as_str()),
            Some(request::Kind::Stop(_) | request::Kind::Nodes(_)) | None => None,
        };
        let forwarded = ToMember {
            kind: Some(to_member::Kind::Forwarded(Forwarded {
                connection: asker.wire(),
                owner: Some(owner.into()),
                request: Some(request.clone()),
            })),
        };
        match target {
            None => {
                // A member whose connection fails is lost: its processes died
                // with it, and the receive that follows removes it.
                let awaiting: Vec<u32> = self
                    .members
                    .iter_mut()
                    .filter_map(|member| {
                        member
                            .connection
                            .deliver(&forwarded)
                            .is_ok()
                            .then_some(member.node_rank)
                    })
                    .collect();
                // Only a stop and an empty request name no node, and both are answered at once: the
                // one request answered later, a preparation, names its node.
                let reply = self.local.reply(owner, asker, request)?;
                if awaiting.is_empty() {
                    return Some(reply);
                }
                self.stops
                    .insert((asker, request.id), PendingStop { reply, awaiting });
                None
            }
            Some(target) if target == self.host => self.local.reply(owner, asker, request),
            Some(target) => match self.members.iter_mut().find(|member| member.host == target) {
                Some(member) => {
                    if member.connection.deliver(&forwarded).is_ok() {
                        member.forwarded.push((asker, request.id));
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
}

/// A stop the leader applied on its own node, awaiting the members' replies.
struct PendingStop {
    /// This node's reply, carrying the last error a member reports.
    reply: Reply,
    /// The node ranks of the members that have not replied yet.
    awaiting: Vec<u32>,
}

/// Runs a member's service until the leader's connection closes or `stop`.
/// Returns what it served when `stop` asks for a hot restart's handoff, and
/// `None` once it stopped.
pub(crate) fn serve_member(
    mut local: Local,
    mut leader: Connection,
    stop: &mut Stop,
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
                            connection,
                            owner,
                            request: Some(request),
                        })),
                })) => {
                    let asker = Asker::of_wire(connection);
                    let reply = match Owner::try_from(owner) {
                        // A preparation is answered once it ends, below.
                        Ok(owner) => local.reply(owner, asker, &request),
                        Err(error) => Some(failed(&request, error)),
                    };
                    if let Some(reply) = reply {
                        leader.deliver(&FromMember {
                            kind: Some(from_member::Kind::Replied(Replied {
                                connection,
                                reply: Some(reply),
                            })),
                        })?;
                    }
                }
                Ok(Some(ToMember {
                    kind: Some(to_member::Kind::Released(released)),
                })) => local
                    .supervisor
                    .release(released.connection, released.engine),
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(error) => {
                    tracing::info!(%error, "the leader's connection closed; ending this node's processes");
                    local.supervisor.terminate_all();
                    return Ok(None);
                }
            }
        }
        for (asker, exit) in local.supervisor.exits() {
            leader.deliver(&FromMember {
                kind: Some(from_member::Kind::Exited(Exited {
                    connection: asker.wire(),
                    exit: Some(exit),
                })),
            })?;
        }
        for (asker, reply) in local.prepared() {
            leader.deliver(&FromMember {
                kind: Some(from_member::Kind::Replied(Replied {
                    connection: asker.wire(),
                    reply: Some(reply),
                })),
            })?;
        }
        // The leader's bytes or close, a process's exit, an ended preparation or a stop end the
        // wait.
        let mut descriptors = leader.poll_fds().to_vec();
        descriptors.push(stop.waker.poll_fd());
        descriptors.extend(local.descriptors().map(readable));
        service::wait_ready(&mut descriptors, None)?;
        stop.waker.drain();
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
        /// Each connection of the process socket, after the listeners.
        peers: Vec<PeerHandoff>,
        next_connection: u64,
        /// Each member's connection, after the peers'.
        members: Vec<MemberHandoff>,
        /// The deployment's nodes (`proto::process::Node`, encoded).
        nodes: Vec<Vec<u8>>,
        /// Each pending stop: its asker and request, this node's reply (encoded) and the members
        /// that have not replied.
        stops: Vec<(Asker, u64, Vec<u8>, Vec<u32>)>,
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

/// A connection of the process socket, its descriptor travelling apart: its identity and the role
/// its hello gave it, which the successor keeps.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct PeerHandoff {
    id: u64,
    role: Role,
    connection: ConnectionHandoff,
}

/// A member's connection, its descriptor travelling apart.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct MemberHandoff {
    node_rank: u32,
    host: String,
    address: String,
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
            peers,
            next_connection,
            // The links follow from the roles that are handed over.
            links: _,
            stops,
            pending: _,
            // The updater's ends are the daemon's, not the service's: the successor makes its own
            // when it resumes (`ProcessCommands`).
            updater: _,
        } = self;
        let (listener_descriptor, endpoint) = listener.hand_over();
        let mut descriptors = vec![listener_descriptor];
        let joins = joins.map(|joins| {
            let (descriptor, endpoint) = joins.listener.hand_over();
            descriptors.push(descriptor);
            endpoint
        });
        let mut peer_handoffs = Vec::with_capacity(peers.len());
        for (id, peer) in peers {
            let connection = peer.connection.hand_over()?;
            peer_handoffs.push(PeerHandoff {
                id,
                role: peer.role,
                connection: ConnectionHandoff::of(&connection),
            });
            descriptors.push(connection.descriptor);
        }
        let mut member_handoffs = Vec::with_capacity(members.len());
        for member in members {
            let connection = member.connection.hand_over()?;
            member_handoffs.push(MemberHandoff {
                node_rank: member.node_rank,
                host: member.host,
                address: member.address,
                connection: ConnectionHandoff::of(&connection),
            });
            descriptors.push(connection.descriptor);
        }
        let local = local.hand_over(&mut descriptors)?;
        Ok((
            ServiceHandoff::Leader {
                host,
                listener: endpoint,
                joins,
                peers: peer_handoffs,
                next_connection,
                members: member_handoffs,
                nodes: nodes.iter().map(prost::Message::encode_to_vec).collect(),
                stops: stops
                    .into_iter()
                    .map(|((asker, request), pending)| {
                        (
                            asker,
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

/// Goes on with the service a predecessor handed over, with its descriptors in the handoff's
/// order, until `stop`: on the leader with the updater's ends `updater`.
pub(crate) fn resume(
    host: String,
    handoff: ServiceHandoff,
    descriptors: Vec<OwnedFd>,
    updater: Updater,
    stop: &mut Stop,
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
            peers,
            next_connection,
            members,
            nodes,
            stops,
            local,
        } => {
            let listener = Listener::resume(next()?, &endpoint)?;
            let joins = joins
                .map(|endpoint| anyhow::Ok(Joins::new(Listener::resume(next()?, &endpoint)?)))
                .transpose()?;
            let mut adopted = HashMap::new();
            let mut links = Links::default();
            for handed in peers {
                links.insert(handed.role, handed.id);
                adopted.insert(
                    handed.id,
                    Peer {
                        connection: handed.connection.resume(next()?)?,
                        role: handed.role,
                    },
                );
            }
            let mut adopted_members = Vec::with_capacity(members.len());
            for member in members {
                adopted_members.push(Membership {
                    node_rank: member.node_rank,
                    host: member.host,
                    address: member.address,
                    connection: member.connection.resume(next()?)?,
                    forwarded: Vec::new(),
                });
            }
            let mut rest: std::vec::IntoIter<OwnedFd> = descriptors.collect::<Vec<_>>().into_iter();
            let local = Local::adopt(host, local, &mut rest, stop.waker.wake())?;
            let nodes = nodes
                .iter()
                .map(|node| Node::decode(node.as_slice()))
                .collect::<Result<Vec<_>, _>>()?;
            let stops = stops
                .into_iter()
                .map(|(asker, request, reply, awaiting)| {
                    Ok((
                        (asker, request),
                        PendingStop {
                            reply: Reply::decode(reply.as_slice())?,
                            awaiting,
                        },
                    ))
                })
                .collect::<anyhow::Result<HashMap<_, _>>>()?;
            tracing::info!(
                connections = adopted.len(),
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
                    peers: adopted,
                    next_connection,
                    links,
                    stops,
                    pending: HashMap::new(),
                    updater,
                },
                stop,
            )
        }
        ServiceHandoff::Member { leader, local } => {
            let leader = leader.resume(next()?)?;
            let mut rest: std::vec::IntoIter<OwnedFd> = descriptors.collect::<Vec<_>>().into_iter();
            let local = Local::adopt(host, local, &mut rest, stop.waker.wake())?;
            tracing::info!("the service goes on from the previous daemon");
            serve_member(local, leader, stop)
        }
    }
}

/// Why a hello is refused, when it is: another version of the engine
/// interface than this daemon's ([`proto::ENGINE_INTERFACE_VERSION`]), or a
/// peer of the wrong kind for the socket it reached.
pub(crate) fn refusal(hello: &Hello, accepts: impl Fn(&hello::Peer) -> bool) -> Option<Refusal> {
    Refusal::of_version(hello.major).or_else(|| match &hello.peer {
        Some(peer) if accepts(peer) => None,
        _ => Some(Refusal::because(Reason::Unserved(Unserved {}))),
    })
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

/// A `pollfd` that waits for `descriptor` to turn readable.
pub(crate) fn readable(descriptor: RawFd) -> libc::pollfd {
    libc::pollfd {
        fd: descriptor,
        events: libc::POLLIN,
        revents: 0,
    }
}
