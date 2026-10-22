//! The engine's client of the process interface: how its workers are started
//! and supervised on every node.
//!
//! The engine starts no process itself. It connects to the leader daemon's
//! process socket and asks it to start each worker on the worker's node
//! (`Spawn`), to hold the store socket a group's first rank serves on that
//! rank's node until the rank inherits it (`Reserve`), to claim the
//! deployment at each node's Storage Server (`Claim`), and to stop a group's
//! workers on every node ahead of the group's next launch (`Stop`). The
//! daemons report each worker's exit; a worker that registered dies when its
//! connection to the engine closes, and one that has not yet registered is
//! known lost by its exit report.
//!
//! One connection serves every group of the deployment and is shared by
//! their threads ([`Processes`]). A request waits for its answer; exits that
//! arrive meanwhile are kept for their groups. Each launch of a group has a
//! generation, which each spawn names and each exit report echoes, so an exit
//! of a launch a stop ended is never attributed to the relaunch.

use std::collections::{BTreeSet, HashMap};
use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use proto::Connection;
use proto::process::{
    Claim, Downstream, Exit, Hello, Node, Request, Reserve, Scope, Spawn, Stop, Upstream, Welcome,
    downstream, event, hello, reply, request, upstream,
};

/// Longest the leader takes to answer a request the engine sends: a claim
/// may start a node's Storage Server, which takes up to 30 seconds.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(90);

/// The deployment's connection to its daemons, shared by its worker groups.
pub(crate) type Processes = Arc<Mutex<ProcessClient>>;

/// Locks the deployment's connection for one operation.
pub(crate) fn lock(
    processes: &Processes,
) -> anyhow::Result<std::sync::MutexGuard<'_, ProcessClient>> {
    processes
        .lock()
        .map_err(|_| anyhow::anyhow!("the process interface connection is poisoned"))
}

/// A listening socket a node's daemon holds for a group.
pub(crate) struct Reservation {
    /// `address:port` processes connect to.
    pub(crate) address: String,
    /// Names the reservation in the spawn of the process that inherits it.
    pub(crate) id: u64,
    /// The descriptor number that process finds the socket at.
    pub(crate) descriptor: i32,
}

/// The exits of each group's running launches.
///
/// A group may run two launches at once: its serving ranks and a standby
/// launched under a later generation beside them.
#[derive(Default)]
struct ExitReports {
    /// The generations of each group's running launches: those its spawns
    /// named, until a stop ends them all.
    running: HashMap<String, BTreeSet<u64>>,
    /// Exits of the running launches, by group and generation, then rank.
    reported: HashMap<(String, u64), HashMap<u32, Exit>>,
}

impl ExitReports {
    /// Records that `group`'s workers are launched under `generation`.
    fn launch(&mut self, group: &str, generation: u64) {
        self.running
            .entry(group.to_owned())
            .or_default()
            .insert(generation);
    }

    /// Keeps one exit for its launch, or discards it when it names a launch
    /// that is not running.
    fn file(&mut self, exit: Exit) {
        let running = self
            .running
            .get(&exit.group)
            .is_some_and(|generations| generations.contains(&exit.generation));
        if !running {
            tracing::debug!(group = %exit.group, rank = exit.rank, generation = exit.generation, "discarding the exit of a stopped worker");
            return;
        }
        self.reported
            .entry((exit.group.clone(), exit.generation))
            .or_default()
            .insert(exit.rank, exit);
    }

    /// The exit of rank `rank` of `group`'s running launch `generation`.
    fn get(&self, group: &str, generation: u64, rank: u32) -> Option<&Exit> {
        self.reported
            .get(&(group.to_owned(), generation))
            .and_then(|ranks| ranks.get(&rank))
    }

    /// Ends every running launch of `group`, or only its launch
    /// `generation`: their kept exits are discarded, as is every exit of
    /// them read later.
    fn stop(&mut self, group: &str, generation: Option<u64>) {
        match generation {
            None => {
                self.running.remove(group);
                self.reported.retain(|(name, _), _| name != group);
            }
            Some(generation) => {
                if let Some(running) = self.running.get_mut(group) {
                    running.remove(&generation);
                }
                self.reported
                    .retain(|(name, launched), _| name != group || *launched != generation);
            }
        }
    }
}

/// The engine's connection to the leader daemon.
pub(crate) struct ProcessClient {
    connection: Connection,
    /// Every node of the deployment, as the leader knows it.
    nodes: Vec<Node>,
    next_request: u64,
    exits: ExitReports,
}

impl ProcessClient {
    /// Connects to the leader daemon's process socket at `address` as an
    /// engine of `deployment`.
    ///
    /// # Errors
    ///
    /// Fails when nothing listens there or the daemon refuses the engine.
    pub(crate) fn connect(address: &str, deployment: &str) -> anyhow::Result<Processes> {
        let mut connection = Connection::connect(address)
            .with_context(|| format!("connecting to the daemon at {address}"))?;
        connection.send(&Hello {
            major: proto::PROTOCOL_MAJOR,
            minor: proto::PROTOCOL_MINOR,
            build: proto::BUILD.to_owned(),
            peer: Some(hello::Peer::Engine(proto::process::Engine {
                deployment: deployment.to_owned(),
            })),
        })?;
        let welcome: Welcome = connection
            .recv(REQUEST_TIMEOUT)?
            .context("the daemon did not answer the engine's hello")?;
        anyhow::ensure!(
            welcome.refusal.is_empty(),
            "the daemon refused the engine: {}",
            welcome.refusal
        );
        tracing::info!(daemon = %address, nodes = welcome.nodes.len(), build = %welcome.build, "connected to the deployment's daemons");
        Ok(Arc::new(Mutex::new(Self {
            connection,
            nodes: welcome.nodes,
            next_request: 1,
            exits: ExitReports::default(),
        })))
    }

    /// The address other nodes reach node `host` at; `None` for a
    /// deployment of one node.
    pub(crate) fn address(&self, host: &str) -> anyhow::Result<Option<IpAddr>> {
        let node = self
            .nodes
            .iter()
            .find(|node| node.host == host)
            .with_context(|| format!("no node of the deployment is host {host}"))?;
        if node.address.is_empty() {
            return Ok(None);
        }
        node.address
            .parse()
            .map(Some)
            .with_context(|| format!("the address of host {host} is {}", node.address))
    }

    /// Starts one worker as `spawn` describes; its group's exits of that
    /// generation are kept from here on.
    ///
    /// # Errors
    ///
    /// Fails when the daemon cannot start it.
    pub(crate) fn spawn(&mut self, spawn: Spawn) -> anyhow::Result<()> {
        self.exits.launch(&spawn.group, spawn.generation);
        let (group, rank, host) = (spawn.group.clone(), spawn.rank, spawn.host.clone());
        self.request(request::Kind::Spawn(spawn))
            .with_context(|| format!("starting rank {rank} of worker group {group} on host {host}"))
            .map(|_| ())
    }

    /// Has node `host`'s daemon bind a socket for `group`, routable from
    /// other nodes when `routable`.
    ///
    /// # Errors
    ///
    /// Fails when the daemon cannot bind one.
    pub(crate) fn reserve(
        &mut self,
        host: &str,
        group: &str,
        routable: bool,
    ) -> anyhow::Result<Reservation> {
        match self.request(request::Kind::Reserve(Reserve {
            host: host.to_owned(),
            group: group.to_owned(),
            routable,
        }))? {
            reply::Kind::Reserved(reserved) => Ok(Reservation {
                address: reserved.address,
                id: reserved.reservation,
                descriptor: reserved.descriptor,
            }),
            other => anyhow::bail!("the daemon answered a reservation with {other:?}"),
        }
    }

    /// Has node `host`'s daemon claim `deployment`'s `scopes` there and
    /// returns the claim's storage launch.
    ///
    /// # Errors
    ///
    /// Fails when the node's Storage Server cannot be started or refuses the
    /// claim.
    pub(crate) fn claim(
        &mut self,
        host: &str,
        deployment: &str,
        scopes: Vec<(String, u32)>,
        server_arguments: Vec<String>,
    ) -> anyhow::Result<ipc::StorageLaunch> {
        let claimed = match self
            .request(request::Kind::Claim(Claim {
                host: host.to_owned(),
                deployment: deployment.to_owned(),
                scopes: scopes
                    .into_iter()
                    .map(|(group, rank)| Scope { group, rank })
                    .collect(),
                server_arguments,
            }))
            .with_context(|| format!("claiming the deployment's storage on host {host}"))?
        {
            reply::Kind::Claimed(claimed) => claimed,
            other => anyhow::bail!("the daemon answered a claim with {other:?}"),
        };
        Ok(ipc::StorageLaunch {
            socket: claimed.socket,
            deployment: deployment.to_owned(),
            claim: claimed
                .claim
                .try_into()
                .map_err(|_| anyhow::anyhow!("a claim token is not 16 bytes"))?,
            // The engine states whether its workers use the host KV cache.
            host_kv_cache: false,
            host_kv_cache_bytes: None,
            tcp: (!claimed.tcp.is_empty()).then_some(claimed.tcp),
            // The engine states whether a group stands by, and how.
            standby: false,
            standby_memory: proto::request::StandbyMemory::Auto,
        })
    }

    /// Stops `group`'s workers on every node ahead of its next launch. The
    /// group's launch ends first, even if the stop cannot be sent: its exits
    /// are discarded, and so is every later report of them.
    ///
    /// # Errors
    ///
    /// Fails when the daemon cannot be reached.
    pub(crate) fn stop(&mut self, group: &str) -> anyhow::Result<()> {
        self.exits.stop(group, None);
        self.request(request::Kind::Stop(Stop {
            group: group.to_owned(),
            generation: None,
        }))
        .map(|_| ())
    }

    /// Stops `group`'s launch `generation` on every node, leaving its other
    /// launches (a standby beside it) running.
    pub(crate) fn stop_generation(&mut self, group: &str, generation: u64) -> anyhow::Result<()> {
        self.exits.stop(group, Some(generation));
        self.request(request::Kind::Stop(Stop {
            group: group.to_owned(),
            generation: Some(generation),
        }))
        .map(|_| ())
    }

    /// The exit status of rank `rank` of `group`'s launch `generation`,
    /// when its daemon reported it. Reads what the connection holds and
    /// never waits.
    pub(crate) fn exited(&mut self, group: &str, generation: u64, rank: u32) -> Option<String> {
        self.pump();
        self.exits
            .get(group, generation, rank)
            .map(|exit| format!("{} on host {}", exit.status, exit.host))
    }

    /// Files the exits the connection holds, without waiting.
    fn pump(&mut self) {
        while let Ok(Some(message)) = self.connection.try_recv::<Downstream>() {
            self.receive(message);
        }
    }

    /// Files an event, or returns a reply.
    fn receive(&mut self, message: Downstream) -> Option<proto::process::Reply> {
        match message.kind {
            Some(downstream::Kind::Reply(reply)) => Some(reply),
            Some(downstream::Kind::Event(event)) => {
                if let Some(event::Kind::Exit(exit)) = event.kind {
                    tracing::warn!(group = %exit.group, rank = exit.rank, host = %exit.host, status = %exit.status, "a worker exited");
                    self.exits.file(exit);
                }
                None
            }
            None => None,
        }
    }

    /// Sends one request and waits for its answer, filing the exits that
    /// arrive meanwhile.
    fn request(&mut self, kind: request::Kind) -> anyhow::Result<reply::Kind> {
        let id = self.next_request;
        self.next_request += 1;
        self.connection.send(&Upstream {
            kind: Some(upstream::Kind::Request(Request {
                id,
                kind: Some(kind),
            })),
        })?;
        let deadline = Instant::now() + REQUEST_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            anyhow::ensure!(
                !remaining.is_zero(),
                "the daemon did not answer within {REQUEST_TIMEOUT:?}"
            );
            let Some(message) = self
                .connection
                .recv::<Downstream>(remaining)
                .context("the daemon's connection failed")?
            else {
                continue;
            };
            if let Some(reply) = self.receive(message) {
                anyhow::ensure!(
                    reply.id == id,
                    "the daemon answered request {} out of order",
                    reply.id
                );
                anyhow::ensure!(reply.error.is_empty(), "{}", reply.error);
                return reply.kind.context("the daemon's answer is empty");
            }
        }
    }
}
