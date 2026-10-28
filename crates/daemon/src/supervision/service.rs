//! The daemon's service thread: the process interface it serves, on the
//! leader to engines and other nodes' daemons, on every other node to the
//! leader.
//!
//! The leader answers an engine's request for its own node itself and
//! forwards a request for another node to that node's daemon, tagged with
//! the engine connection it serves; the answer and the node's exits come
//! back the same way. A member daemon serves the leader's forwarded requests
//! until the leader's connection closes, and then ends its processes.
//!
//! One thread does all of it, waiting on every connection and a wake pipe
//! at once and polling its children for exits at a short interval, so no
//! request waits behind another node's.
//!
//! A hot restart of the daemon stops the thread between two rounds and
//! hands everything it serves to the successor daemon: its listeners and
//! connections as descriptors, with the bytes each received and has not
//! taken, its pending stops, its processes and its claims
//! ([`HandedService`]). The successor's thread goes on from there; no peer
//! sees a connection close.

use std::collections::HashMap;
use std::net::IpAddr;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use prost::Message as _;
use proto::process::{
    self, Answered, Downstream, Event, Exited, Forwarded, FromMember, Hello, LedgerReport, Node,
    NodesKnown, Reply, Request, Spawned, Stopped, ToMember, TransfersThrottled, Upstream, Welcome,
    downstream, event, from_member, hello, reply, request, to_member, upstream,
};
use proto::{Connection, Handed, Listener};

use crate::supervision::claims::{ClaimHolder, HandedClaims};
use crate::supervision::supervisor::{HandedProcesses, Supervisor};

/// How often children are polled for exits while nothing else happens.
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
}

impl Local {
    /// Hands the supervisor's processes and the claims over: their state,
    /// and their descriptors appended to `descriptors`.
    fn hand_over(&mut self, descriptors: &mut Vec<OwnedFd>) -> anyhow::Result<HandedLocal> {
        let (claims, claim_descriptors) = self.claims.hand_over()?;
        let (processes, process_descriptors) = self.supervisor.hand_over()?;
        let handed = HandedLocal {
            claims,
            claim_descriptors: claim_descriptors.len(),
            processes,
            process_descriptors: process_descriptors.len(),
            address: self.address,
        };
        descriptors.extend(claim_descriptors);
        descriptors.extend(process_descriptors);
        Ok(handed)
    }

    /// Goes on with what [`Local::hand_over`] handed, taking its
    /// descriptors from `descriptors`.
    fn adopt(
        host: String,
        handed: HandedLocal,
        descriptors: &mut std::vec::IntoIter<OwnedFd>,
    ) -> anyhow::Result<Self> {
        let claims: Vec<OwnedFd> = descriptors
            .by_ref()
            .take(handed.claim_descriptors)
            .collect();
        let processes: Vec<OwnedFd> = descriptors
            .by_ref()
            .take(handed.process_descriptors)
            .collect();
        anyhow::ensure!(
            claims.len() == handed.claim_descriptors
                && processes.len() == handed.process_descriptors,
            "the handed-over service lacks descriptors"
        );
        Ok(Self {
            supervisor: Supervisor::adopt(host, handed.processes, processes)?,
            claims: ClaimHolder::adopt(handed.address, handed.claims, claims),
            address: handed.address,
        })
    }

    /// Answers one request for this node on behalf of `engine`.
    fn answer(&mut self, engine: u64, request: &Request) -> Reply {
        let mut reply = Reply {
            id: request.id,
            ..Reply::default()
        };
        match &request.kind {
            Some(request::Kind::Spawn(spawn)) => match self.supervisor.spawn(engine, spawn) {
                Ok(pid) => reply.kind = Some(reply::Kind::Spawned(Spawned { pid })),
                Err(error) => reply.error = error,
            },
            Some(request::Kind::Stop(stop)) => {
                self.supervisor.stop(engine, &stop.group, stop.generation);
                reply.kind = Some(reply::Kind::Stopped(Stopped {}));
            }
            Some(request::Kind::Reserve(reserve)) => {
                match self.supervisor.reserve(engine, reserve, self.address) {
                    Ok(reserved) => reply.kind = Some(reply::Kind::Reserved(reserved)),
                    Err(error) => reply.error = error,
                }
            }
            Some(request::Kind::Claim(claim)) => match self.claims.claim(claim) {
                Ok(claimed) => reply.kind = Some(reply::Kind::Claimed(claimed)),
                Err(error) => reply.error = format!("{error:#}"),
            },
            Some(request::Kind::PublishWorlds(publish)) => {
                match self
                    .claims
                    .publish_worlds(&publish.deployment, &publish.worlds)
                {
                    Ok(()) => {
                        reply.kind = Some(reply::Kind::WorldsPublished(
                            proto::process::WorldsPublished {},
                        ))
                    }
                    Err(error) => reply.error = format!("{error:#}"),
                }
            }
            Some(request::Kind::KeepWeights(keep)) => {
                match self.claims.keep_weights(
                    &keep.deployment,
                    &keep.worlds,
                    keep.to_host,
                    &keep.cache_keys,
                ) {
                    Ok(kept) => reply.kind = Some(reply::Kind::WeightsKept(kept)),
                    Err(error) => reply.error = format!("{error:#}"),
                }
            }
            Some(request::Kind::Ledger(query)) => match self.claims.ledger(&query.deployment) {
                Ok(ledgers) => reply.kind = Some(reply::Kind::Ledger(LedgerReport { ledgers })),
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
                reply.error = "the leader answers for the deployment's nodes".to_owned();
            }
            Some(request::Kind::Devices(_)) => match available_devices() {
                Ok(available) => {
                    reply.kind = Some(reply::Kind::Devices(process::DeviceInventory { available }))
                }
                Err(error) => reply.error = format!("{error:#}"),
            },
            None => reply.error = "an empty request".to_owned(),
        }
        reply
    }
}

fn available_devices() -> anyhow::Result<Vec<u32>> {
    let nvml = cuda::Nvml::init()?;
    let mut available = Vec::new();
    for ordinal in 0..cuda::DeviceId::count()? {
        if nvml.is_idle(cuda::DeviceId(ordinal))? {
            available.push(ordinal as u32);
        }
    }
    Ok(available)
}

/// A member daemon as the leader holds it.
pub(crate) struct Membership {
    pub(crate) node_rank: u32,
    pub(crate) host: String,
    pub(crate) connection: Connection,
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

    /// Accepts the connections waiting and answers every hello that
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
            let welcome = Welcome {
                major: proto::PROTOCOL_MAJOR,
                minor: proto::PROTOCOL_MINOR,
                build: proto::BUILD.to_owned(),
                refusal: refusal.clone().unwrap_or_default(),
                nodes: Vec::new(),
            };
            let sent = connection.send(&welcome).is_ok();
            if let (None, Some(member), true) = (refusal, member, sent) {
                tracing::info!(node_rank = member.node_rank, host = %member.host, "a node joined");
                joined.push(Membership {
                    node_rank: member.node_rank,
                    host: member.host,
                    connection,
                });
            }
        }
        Ok(joined)
    }

    /// The descriptors a wait for joins polls.
    fn descriptors(&self) -> impl Iterator<Item = std::os::fd::RawFd> + '_ {
        std::iter::once(self.listener.descriptor())
            .chain(self.connecting.iter().map(Connection::descriptor))
    }
}

/// The deployment's node of `member`, as engines learn it.
fn member_node(member: &Membership) -> Node {
    Node {
        node_rank: member.node_rank,
        host: member.host.clone(),
        address: member
            .connection
            .peer_ip()
            .map(|address| address.to_string())
            .unwrap_or_default(),
    }
}

/// An engine connection to the leader.
struct EngineConnection {
    connection: Connection,
    /// Whether its hello was answered.
    welcomed: bool,
}

/// Runs the leader's service until `stop`: engines connect on `listener`,
/// `members` joined before, and other nodes may join at `joins`. Returns
/// what it served when `stop` asks for a hot restart's handoff, and `None`
/// once it stopped.
pub(crate) fn serve_leader(
    host: &str,
    local: Local,
    listener: Listener,
    joins: Option<Joins>,
    members: Vec<Membership>,
    stop: &Stop,
) -> anyhow::Result<Option<(HandedService, Vec<OwnedFd>)>> {
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
            stops: HashMap::new(),
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
    /// Stops of an engine, by its request id, awaiting the members'
    /// answers.
    stops: HashMap<(u64, u64), PendingStop>,
}

/// Runs the leader's service from `service` until `stop`.
pub(crate) fn run_leader(
    service: LeaderService,
    stop: &Stop,
) -> anyhow::Result<Option<(HandedService, Vec<OwnedFd>)>> {
    let LeaderService {
        host,
        mut local,
        listener,
        mut joins,
        mut members,
        mut nodes,
        mut engines,
        mut next_engine,
        mut stops,
    } = service;
    let host = host.as_str();
    loop {
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
                stops,
            };
            return service.hand_over().map(Some);
        }
        if stop.requested() {
            local.supervisor.terminate_all();
            return Ok(None);
        }
        while let Some(connection) = listener.try_accept()? {
            engines.insert(
                next_engine,
                EngineConnection {
                    connection,
                    welcomed: false,
                },
            );
            next_engine += 1;
        }

        // Nodes joining the running deployment: an engine that connects
        // from now on learns them, so the next world may run ranks there.
        if let Some(joins) = &mut joins {
            for member in joins.admit(&members)? {
                nodes.push(member_node(&member));
                nodes.sort_by_key(|node| node.node_rank);
                members.push(member);
            }
        }

        // Engines: their hellos and requests; a closed engine takes its
        // processes on every node with it.
        let mut closed = Vec::new();
        for (&engine, peer) in &mut engines {
            loop {
                if !peer.welcomed {
                    match peer.connection.try_recv::<Hello>() {
                        Ok(Some(hello)) => {
                            let refusal =
                                refusal(&hello, |peer| matches!(peer, hello::Peer::Engine(_)));
                            let welcome = Welcome {
                                major: proto::PROTOCOL_MAJOR,
                                minor: proto::PROTOCOL_MINOR,
                                build: proto::BUILD.to_owned(),
                                refusal: refusal.clone().unwrap_or_default(),
                                nodes: nodes.clone(),
                            };
                            if peer.connection.send(&welcome).is_err() || refusal.is_some() {
                                closed.push(engine);
                                break;
                            }
                            tracing::info!(engine, build = %hello.build, "an engine connected");
                            peer.welcomed = true;
                            continue;
                        }
                        Ok(None) => break,
                        Err(_) => {
                            closed.push(engine);
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
                            &request,
                            host,
                            &mut local,
                            &mut members,
                            &nodes,
                            &mut stops,
                        ) && peer.connection.send(&downstream_reply(reply)).is_err()
                        {
                            closed.push(engine);
                            break;
                        }
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(_) => {
                        closed.push(engine);
                        break;
                    }
                }
            }
        }
        for engine in closed {
            engines.remove(&engine);
            stops.retain(|&(owner, _), _| owner != engine);
            tracing::info!(engine, "an engine disconnected; ending its processes");
            local.supervisor.release(engine);
            for member in &mut members {
                let _ = member.connection.send(&ToMember {
                    kind: Some(to_member::Kind::Released(engine)),
                });
            }
        }

        // Members: answers and exits on their nodes, for their engines.
        let mut lost = Vec::new();
        for member in &mut members {
            loop {
                match member.connection.try_recv::<FromMember>() {
                    Ok(Some(message)) => {
                        let (engine, downstream) = match message.kind {
                            Some(from_member::Kind::Answered(Answered {
                                engine,
                                reply: Some(reply),
                            })) => match stops.get_mut(&(engine, reply.id)) {
                                // A stop is answered once every node stopped
                                // the group.
                                Some(pending) => {
                                    if !reply.error.is_empty() {
                                        pending.reply.error = reply.error;
                                    }
                                    pending.awaiting.retain(|&rank| rank != member.node_rank);
                                    if !pending.awaiting.is_empty() {
                                        continue;
                                    }
                                    let Some(pending) = stops.remove(&(engine, reply.id)) else {
                                        continue;
                                    };
                                    (engine, downstream_reply(pending.reply))
                                }
                                None => (engine, downstream_reply(reply)),
                            },
                            Some(from_member::Kind::Exited(Exited {
                                engine,
                                exit: Some(exit),
                            })) => (engine, downstream_exit(exit)),
                            _ => continue,
                        };
                        if let Some(peer) = engines.get_mut(&engine) {
                            let _ = peer.connection.send(&downstream);
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

        // A lost node runs nothing more: requests for it fail from here on,
        // and the stops that awaited it are answered without it.
        if !lost.is_empty() {
            members.retain(|member| !lost.contains(&member.node_rank));
            nodes.retain(|node| !lost.contains(&node.node_rank));
            let mut settled = Vec::new();
            for (&key, pending) in &mut stops {
                pending.awaiting.retain(|rank| !lost.contains(rank));
                if pending.awaiting.is_empty() {
                    settled.push(key);
                }
            }
            for key in settled {
                if let Some(pending) = stops.remove(&key)
                    && let Some(peer) = engines.get_mut(&key.0)
                {
                    let _ = peer.connection.send(&downstream_reply(pending.reply));
                }
            }
        }

        // This node's exits, to the engines that started the processes.
        for (engine, exit) in local.supervisor.exits() {
            if let Some(peer) = engines.get_mut(&engine) {
                let _ = peer.connection.send(&downstream_exit(exit));
            }
        }

        let mut descriptors = vec![listener.descriptor(), stop.wake.as_raw_fd()];
        descriptors.extend(joins.iter().flat_map(Joins::descriptors));
        descriptors.extend(engines.values().map(|peer| peer.connection.descriptor()));
        descriptors.extend(members.iter().map(|member| member.connection.descriptor()));
        wait(&descriptors, EXIT_POLL_INTERVAL);
        stop.drain();
    }
}

/// A stop the leader applied on its own node, awaiting the members' answers.
struct PendingStop {
    /// This node's answer, carrying the last error a member reports.
    reply: Reply,
    /// The node ranks of the members that have not answered yet.
    awaiting: Vec<u32>,
}

/// Routes one engine request: answered here for this node, forwarded to the
/// node's daemon otherwise, whose answer comes back later. A stop applies to
/// every node and is answered once every node stopped the group: it waits in
/// `stops` for the members' answers.
fn route(
    engine: u64,
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
    let target = match &request.kind {
        Some(request::Kind::Spawn(spawn)) => Some(spawn.host.as_str()),
        Some(request::Kind::Reserve(reserve)) => Some(reserve.host.as_str()),
        Some(request::Kind::Claim(claim)) => Some(claim.host.as_str()),
        Some(request::Kind::Ledger(query)) => Some(query.host.as_str()),
        Some(request::Kind::KeepWeights(keep)) => Some(keep.host.as_str()),
        Some(request::Kind::ThrottleTransfers(throttle)) => Some(throttle.host.as_str()),
        Some(request::Kind::Devices(query)) => Some(query.host.as_str()),
        Some(request::Kind::PublishWorlds(publish)) => Some(publish.host.as_str()),
        Some(request::Kind::Stop(_) | request::Kind::Nodes(_)) | None => None,
    };
    let forward = |member: &mut Membership| {
        member
            .connection
            .send(&ToMember {
                kind: Some(to_member::Kind::Forwarded(Forwarded {
                    engine,
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
            let reply = local.answer(engine, request);
            if awaiting.is_empty() {
                return Some(reply);
            }
            stops.insert((engine, request.id), PendingStop { reply, awaiting });
            None
        }
        Some(target) if target == host => Some(local.answer(engine, request)),
        Some(target) => match members.iter_mut().find(|member| member.host == target) {
            Some(member) => {
                if forward(member) {
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
) -> anyhow::Result<Option<(HandedService, Vec<OwnedFd>)>> {
    loop {
        if stop.handoff() {
            let mut descriptors = Vec::new();
            let local = local.hand_over(&mut descriptors)?;
            let leader = leader.hand_over();
            let handed = HandedService::Member {
                local,
                leader: HandedConnection::of(&leader),
            };
            let mut ordered = vec![leader.descriptor];
            ordered.extend(descriptors);
            return Ok(Some((handed, ordered)));
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
                            request: Some(request),
                        })),
                })) => {
                    let reply = local.answer(engine, &request);
                    leader.send(&FromMember {
                        kind: Some(from_member::Kind::Answered(Answered {
                            engine,
                            reply: Some(reply),
                        })),
                    })?;
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
            leader.send(&FromMember {
                kind: Some(from_member::Kind::Exited(Exited {
                    engine,
                    exit: Some(exit),
                })),
            })?;
        }
        wait(
            &[leader.descriptor(), stop.wake.as_raw_fd()],
            EXIT_POLL_INTERVAL,
        );
        stop.drain();
    }
}

/// What a daemon's service hands its successor in a hot restart: its state,
/// beside the descriptors that travel with it in the order the fields name
/// them.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) enum HandedService {
    Leader {
        host: String,
        /// The process socket: its address and whether it is a Unix
        /// socket; its descriptor comes first.
        listener: (String, bool),
        /// The members' address, when the deployment names one: its
        /// address and whether it is a Unix socket; its descriptor follows
        /// the process socket's. A node whose hello is still awaited joins
        /// again.
        #[serde(default)]
        joins: Option<(String, bool)>,
        /// Each engine connection, by its identity, after the listener.
        engines: Vec<(u64, bool, HandedConnection)>,
        next_engine: u64,
        /// Each member's connection, after the engines'.
        members: Vec<(u32, String, HandedConnection)>,
        /// The deployment's nodes (`proto::process::Node`, encoded).
        nodes: Vec<Vec<u8>>,
        /// Each pending stop: its engine and request, this node's reply
        /// (encoded) and the members that have not answered.
        stops: Vec<(u64, u64, Vec<u8>, Vec<u32>)>,
        /// The claims' and processes' descriptors follow the members'.
        local: HandedLocal,
    },
    Member {
        /// The leader's connection, whose descriptor comes first.
        leader: HandedConnection,
        local: HandedLocal,
    },
}

/// A connection's state, its descriptor travelling apart.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct HandedConnection {
    unix: bool,
    received: Vec<u8>,
}

impl HandedConnection {
    fn of(handed: &Handed) -> Self {
        Self {
            unix: handed.unix,
            received: handed.received.clone(),
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

/// The supervisor's processes and the node's claims, as handed over.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct HandedLocal {
    claims: HandedClaims,
    claim_descriptors: usize,
    processes: HandedProcesses,
    process_descriptors: usize,
    address: Option<IpAddr>,
}

impl LeaderService {
    /// The handoff of everything the service holds.
    fn hand_over(self) -> anyhow::Result<(HandedService, Vec<OwnedFd>)> {
        let Self {
            host,
            mut local,
            listener,
            joins,
            members,
            nodes,
            engines,
            next_engine,
            stops,
        } = self;
        let (listener_descriptor, unix, address) = listener.hand_over();
        let mut descriptors = vec![listener_descriptor];
        let joins = joins.map(|joins| {
            let (descriptor, unix, address) = joins.listener.hand_over();
            descriptors.push(descriptor);
            (address, unix)
        });
        let mut handed_engines = Vec::with_capacity(engines.len());
        for (id, engine) in engines {
            let handed = engine.connection.hand_over();
            handed_engines.push((id, engine.welcomed, HandedConnection::of(&handed)));
            descriptors.push(handed.descriptor);
        }
        let mut handed_members = Vec::with_capacity(members.len());
        for member in members {
            let handed = member.connection.hand_over();
            handed_members.push((member.node_rank, member.host, HandedConnection::of(&handed)));
            descriptors.push(handed.descriptor);
        }
        let local = local.hand_over(&mut descriptors)?;
        Ok((
            HandedService::Leader {
                host,
                listener: (address, unix),
                joins,
                engines: handed_engines,
                next_engine,
                members: handed_members,
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
pub(crate) fn resume(
    host: String,
    handed: HandedService,
    descriptors: Vec<OwnedFd>,
    stop: &Stop,
) -> anyhow::Result<Option<(HandedService, Vec<OwnedFd>)>> {
    let mut descriptors = descriptors.into_iter();
    let mut next = || {
        descriptors
            .next()
            .ok_or_else(|| anyhow::anyhow!("the handed-over service lacks a descriptor"))
    };
    match handed {
        HandedService::Leader {
            host: leader_host,
            listener: (address, unix),
            joins,
            engines,
            next_engine,
            members,
            nodes,
            stops,
            local,
        } => {
            let listener = Listener::resume(next()?, unix, address)?;
            let joins = joins
                .map(|(address, unix)| {
                    anyhow::Ok(Joins::new(Listener::resume(next()?, unix, address)?))
                })
                .transpose()?;
            let mut adopted_engines = HashMap::new();
            for (id, welcomed, connection) in engines {
                adopted_engines.insert(
                    id,
                    EngineConnection {
                        connection: connection.resume(next()?)?,
                        welcomed,
                    },
                );
            }
            let mut adopted_members = Vec::with_capacity(members.len());
            for (node_rank, member_host, connection) in members {
                adopted_members.push(Membership {
                    node_rank,
                    host: member_host,
                    connection: connection.resume(next()?)?,
                });
            }
            let mut rest: std::vec::IntoIter<OwnedFd> = descriptors.collect::<Vec<_>>().into_iter();
            let local = Local::adopt(host, local, &mut rest)?;
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
                    stops,
                },
                stop,
            )
        }
        HandedService::Member { leader, local } => {
            let leader = leader.resume(next()?)?;
            let mut rest: std::vec::IntoIter<OwnedFd> = descriptors.collect::<Vec<_>>().into_iter();
            let local = Local::adopt(host, local, &mut rest)?;
            tracing::info!("the service goes on from the previous daemon");
            serve_member(local, leader, stop)
        }
    }
}

/// Why a hello is refused, when it is: another major version, or a peer of
/// the wrong kind for the socket it reached.
pub(crate) fn refusal(hello: &Hello, accepts: impl Fn(&hello::Peer) -> bool) -> Option<String> {
    if hello.major != proto::PROTOCOL_MAJOR {
        return Some(format!(
            "the peer speaks major version {} and this daemon {}",
            hello.major,
            proto::PROTOCOL_MAJOR
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
