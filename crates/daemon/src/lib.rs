//! The daemon: the persistent process of one node of a deployment.
//!
//! Every node runs one daemon (`uniserve serve`); the daemons are peers, and
//! the leader is the node the deployment is started on (node rank 0). A
//! daemon supervises the processes of its node (`supervision`): it starts
//! them as its children, reaps them, cleans what a process that did not end
//! in order left in `/dev/shm`, and reports their exits. It holds its node's
//! claim at the Storage Server (`claims`), so a claim outlives every engine.
//!
//! The daemons talk the process interface (`proto::process`). Each member
//! daemon joins the leader at the leader's address; the leader then accepts
//! engines on its process socket, answers their requests for its own node
//! and forwards the others to the member that owns the node (`service`).

mod claims;
mod service;
mod supervision;

use std::net::IpAddr;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use proto::process::{Hello, Join, Welcome, hello};
use proto::{Connection, Listener};

use crate::claims::ClaimHolder;
use crate::service::{Local, Stop};
use crate::supervision::Supervisor;

/// How long the leader waits for every other node's daemon to join, and a
/// member to reach the leader: the nodes of a deployment are started
/// together, so this covers one that is starting rather than one waiting to
/// be scheduled.
pub const JOIN_TIMEOUT: Duration = Duration::from_secs(300);

/// Where a daemon runs and how it reaches the other nodes' daemons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonOptions {
    /// Nodes of the deployment.
    pub nnodes: u32,
    /// This node's rank; the leader is rank 0.
    pub node_rank: u32,
    /// This node's host identity, as placements name it.
    pub host: String,
    /// The leader's `host:port` the members join at; the leader binds it.
    /// Required with several nodes.
    pub dist_init_addr: Option<String>,
    /// The leader's process socket, which engines connect to.
    pub socket: PathBuf,
    /// The `uniserve` executable that starts this node's Storage Server
    /// when it has none, as its `storage serve` command.
    pub executable: PathBuf,
    /// How long the leader waits for the members and a member for the
    /// leader.
    pub join_timeout: Duration,
}

/// A running daemon. Dropping it stops its service and ends every process
/// it supervises.
pub struct Daemon {
    /// The process socket engines connect to, on the leader.
    address: Option<String>,
    /// The leader's socket file, removed once its service ends.
    socket: Option<PathBuf>,
    requested: Arc<AtomicBool>,
    wake: UnixStream,
    thread: Option<JoinHandle<anyhow::Result<()>>>,
}

impl Daemon {
    /// Starts this node's daemon. The leader binds its process socket and
    /// its members' address and waits until every member joined; a member
    /// joins the leader. Returns once the service runs.
    ///
    /// # Errors
    ///
    /// Fails when a socket cannot be bound, a member does not join in time,
    /// or the leader cannot be reached or refuses this member.
    pub fn start(options: DaemonOptions) -> anyhow::Result<Self> {
        anyhow::ensure!(
            options.nnodes >= 1 && options.node_rank < options.nnodes,
            "node rank {} lies outside {} nodes",
            options.node_rank,
            options.nnodes
        );
        anyhow::ensure!(
            options.nnodes == 1 || options.dist_init_addr.is_some(),
            "a deployment of several nodes needs the leader's address (`--dist-init-addr`)"
        );
        let (wake, wake_reader) = UnixStream::pair().context("creating the daemon's wake")?;
        wake_reader.set_nonblocking(true)?;
        let requested = Arc::new(AtomicBool::new(false));
        let stop = Stop {
            requested: Arc::clone(&requested),
            wake: wake_reader,
        };

        let (address, thread) = if options.node_rank == 0 {
            let address = format!("unix:{}", options.socket.display());
            if let Some(parent) = options.socket.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            // A socket a stopped daemon left behind is replaced.
            let _ = std::fs::remove_file(&options.socket);
            let listener = Listener::bind(&address)
                .with_context(|| format!("binding the process socket {address}"))?;
            let members = join_members(&options)?;
            // The leader is reached at the address its members connected to.
            let leader_address = members
                .first()
                .and_then(|member| member.connection.local_ip());
            let local = local(&options, leader_address, false);
            let host = options.host.clone();
            let thread = std::thread::Builder::new()
                .name("daemon".into())
                .spawn(move || service::serve_leader(&host, local, listener, members, &stop))
                .context("starting the daemon's service")?;
            (Some((address, options.socket.clone())), thread)
        } else {
            let leader = join_leader(&options)?;
            let address = leader.local_ip();
            let local = local(&options, address, true);
            let thread = std::thread::Builder::new()
                .name("daemon".into())
                .spawn(move || service::serve_member(local, leader, &stop))
                .context("starting the daemon's service")?;
            (None, thread)
        };
        let (address, socket) = address.unzip();
        Ok(Self {
            address,
            socket,
            requested,
            wake,
            thread: Some(thread),
        })
    }

    /// The leader's process socket, which engines connect to; `None` on a
    /// member.
    pub fn address(&self) -> Option<&str> {
        self.address.as_deref()
    }

    /// Whether the service still runs: a member's ends when the leader's
    /// connection closes.
    pub fn is_running(&self) -> bool {
        self.thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
    }

    /// Waits until the service ends: on a member, when the leader's
    /// connection closes.
    ///
    /// # Errors
    ///
    /// Returns the service's failure.
    pub fn wait(mut self) -> anyhow::Result<()> {
        self.join()
    }

    /// Stops the service and ends every process it supervises.
    pub fn stop(&mut self) -> anyhow::Result<()> {
        use std::io::Write as _;

        self.requested.store(true, Ordering::Release);
        let _ = (&self.wake).write(&[1]);
        self.join()
    }

    fn join(&mut self) -> anyhow::Result<()> {
        let Some(thread) = self.thread.take() else {
            return Ok(());
        };
        let served = thread
            .join()
            .map_err(|_| anyhow::anyhow!("the daemon's service panicked"));
        if let Some(socket) = self.socket.take() {
            let _ = std::fs::remove_file(socket);
        }
        served?
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// This node's supervisor and claims. `remote_pages` says whether another
/// node's processes send this node's Storage Server page requests: a member
/// node's server serves the engine on the leader.
fn local(options: &DaemonOptions, address: Option<IpAddr>, remote_pages: bool) -> Local {
    Local {
        supervisor: Supervisor::new(options.host.clone()),
        claims: ClaimHolder::new(options.executable.clone(), remote_pages, address),
        address,
    }
}

/// The leader's wait for every member: each connects to `dist_init_addr`
/// and says hello.
fn join_members(options: &DaemonOptions) -> anyhow::Result<Vec<service::Membership>> {
    if options.nnodes == 1 {
        return Ok(Vec::new());
    }
    let address = options
        .dist_init_addr
        .as_deref()
        .context("a leader of several nodes needs its address")?;
    let bind = format!(
        "0.0.0.0:{}",
        address
            .rsplit_once(':')
            .map(|(_, port)| port)
            .context("the leader's address names no port")?
    );
    let listener =
        Listener::bind(&bind).with_context(|| format!("binding the leader's address {bind}"))?;
    tracing::info!(address = %listener.address(), nodes = options.nnodes, "awaiting the deployment's other nodes");

    let deadline = Instant::now() + options.join_timeout;
    let mut connecting: Vec<Connection> = Vec::new();
    let mut members: Vec<service::Membership> = Vec::new();
    while members.len() + 1 < options.nnodes as usize {
        anyhow::ensure!(
            Instant::now() < deadline,
            "{} of {} nodes joined within {:?}",
            members.len() + 1,
            options.nnodes,
            options.join_timeout
        );
        while let Some(connection) = listener.try_accept()? {
            connecting.push(connection);
        }
        let mut index = 0;
        while index < connecting.len() {
            let hello = match connecting[index].try_recv::<Hello>() {
                Ok(Some(hello)) => hello,
                Ok(None) => {
                    index += 1;
                    continue;
                }
                Err(_) => {
                    connecting.swap_remove(index);
                    continue;
                }
            };
            let mut connection = connecting.swap_remove(index);
            let joined = match &hello.peer {
                Some(hello::Peer::Join(join)) => Some(join.clone()),
                _ => None,
            };
            let refusal = service::refusal(&hello, |peer| matches!(peer, hello::Peer::Join(_)))
                .or_else(|| {
                    let member = joined.as_ref()?;
                    (member.node_rank == 0
                        || member.node_rank >= options.nnodes
                        || members
                            .iter()
                            .any(|known| known.node_rank == member.node_rank))
                    .then(|| {
                        format!(
                            "node rank {} is not one this deployment awaits",
                            member.node_rank
                        )
                    })
                });
            let welcome = Welcome {
                major: proto::PROTOCOL_MAJOR,
                minor: proto::PROTOCOL_MINOR,
                build: proto::BUILD.to_owned(),
                refusal: refusal.clone().unwrap_or_default(),
                nodes: Vec::new(),
            };
            let sent = connection.send(&welcome).is_ok();
            if let (None, Some(member), true) = (refusal, joined, sent) {
                tracing::info!(node_rank = member.node_rank, host = %member.host, "a node joined");
                members.push(service::Membership {
                    node_rank: member.node_rank,
                    host: member.host,
                    connection,
                });
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(members)
}

/// A member's join: connects to the leader, retrying until the deadline,
/// and says hello.
fn join_leader(options: &DaemonOptions) -> anyhow::Result<Connection> {
    let address = options
        .dist_init_addr
        .as_deref()
        .context("a member needs the leader's address")?;
    let deadline = Instant::now() + options.join_timeout;
    let mut connection = loop {
        match Connection::connect(address) {
            Ok(connection) => break connection,
            Err(error) if Instant::now() < deadline => {
                tracing::debug!(%error, "the leader is not reachable yet");
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(error) => {
                return Err(error).with_context(|| format!("joining the leader at {address}"));
            }
        }
    };
    connection.send(&Hello {
        major: proto::PROTOCOL_MAJOR,
        minor: proto::PROTOCOL_MINOR,
        build: proto::BUILD.to_owned(),
        peer: Some(hello::Peer::Join(Join {
            node_rank: options.node_rank,
            host: options.host.clone(),
        })),
    })?;
    let welcome: Welcome = connection
        .recv(options.join_timeout)?
        .context("the leader did not answer this node's hello")?;
    anyhow::ensure!(
        welcome.refusal.is_empty(),
        "the leader refused this node: {}",
        welcome.refusal
    );
    tracing::info!(leader = %address, node_rank = options.node_rank, "joined the leader");
    Ok(connection)
}
