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

use std::collections::HashMap;
use std::net::IpAddr;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use proto::process::{
    Answered, Downstream, Event, Exited, Forwarded, FromMember, Hello, Node, Reply, Request,
    Spawned, Stopped, ToMember, Upstream, Welcome, downstream, event, from_member, hello, reply,
    request, to_member, upstream,
};
use proto::{Connection, Listener};

use crate::supervision::claims::ClaimHolder;
use crate::supervision::supervisor::Supervisor;

/// How often children are polled for exits while nothing else happens.
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// What the service thread reads its stop from.
pub(crate) struct Stop {
    pub(crate) requested: Arc<AtomicBool>,
    pub(crate) wake: UnixStream,
}

impl Stop {
    fn requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
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
                self.supervisor.stop(engine, &stop.group);
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
            None => reply.error = "an empty request".to_owned(),
        }
        reply
    }
}

/// A member daemon as the leader holds it.
pub(crate) struct Membership {
    pub(crate) node_rank: u32,
    pub(crate) host: String,
    pub(crate) connection: Connection,
}

/// An engine connection to the leader.
struct EngineConnection {
    connection: Connection,
    /// Whether its hello was answered.
    welcomed: bool,
}

/// Runs the leader's service until `stop`: engines connect on `listener`,
/// and `members` joined before.
pub(crate) fn serve_leader(
    host: &str,
    mut local: Local,
    listener: Listener,
    mut members: Vec<Membership>,
    stop: &Stop,
) -> anyhow::Result<()> {
    let mut nodes = vec![Node {
        node_rank: 0,
        host: host.to_owned(),
        address: local
            .address
            .map(|address| address.to_string())
            .unwrap_or_default(),
    }];
    for member in &members {
        nodes.push(Node {
            node_rank: member.node_rank,
            host: member.host.clone(),
            address: member
                .connection
                .peer_ip()
                .map(|address| address.to_string())
                .unwrap_or_default(),
        });
    }
    nodes.sort_by_key(|node| node.node_rank);

    let mut engines: HashMap<u64, EngineConnection> = HashMap::new();
    let mut next_engine = 1_u64;
    // Stops of an engine, by its request id, awaiting the members' answers.
    let mut stops: HashMap<(u64, u64), PendingStop> = HashMap::new();
    loop {
        if stop.requested() {
            local.supervisor.terminate_all();
            return Ok(());
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
                        if let Some(reply) =
                            route(engine, &request, host, &mut local, &mut members, &mut stops)
                            && peer.connection.send(&downstream_reply(reply)).is_err()
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
    stops: &mut HashMap<(u64, u64), PendingStop>,
) -> Option<Reply> {
    let target = match &request.kind {
        Some(request::Kind::Spawn(spawn)) => Some(spawn.host.as_str()),
        Some(request::Kind::Reserve(reserve)) => Some(reserve.host.as_str()),
        Some(request::Kind::Claim(claim)) => Some(claim.host.as_str()),
        Some(request::Kind::Stop(_)) | None => None,
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
pub(crate) fn serve_member(
    mut local: Local,
    mut leader: Connection,
    stop: &Stop,
) -> anyhow::Result<()> {
    loop {
        if stop.requested() {
            local.supervisor.terminate_all();
            return Ok(());
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
                    return Ok(());
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
