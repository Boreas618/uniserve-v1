//! Supervision: what every node's daemon does for the processes of its
//! node.
//!
//! The daemons are peers. A daemon supervises the processes of its node
//! (`supervisor`): it starts them as its children, reaps them, cleans what a
//! process that did not end in order left in `/dev/shm`, and reports their
//! exits. It holds its node's claim at the Storage Server (`claims`), so a
//! claim outlives every engine.
//!
//! The daemons talk the process interface (`proto::process`). Each member
//! daemon joins the leader at the leader's address; the leader then accepts
//! engines on its process socket, answers their requests for its own node
//! and forwards the others to the member that owns the node (`service`).
//!
//! A hot restart hands a running daemon's service to its successor
//! ([`Daemon::hand_over`], [`Daemon::resume`]): the successor goes on with
//! its sockets, connections, processes and claims.

mod claims;

pub(crate) use claims::restart_earlier;
mod service;
mod supervisor;

pub(crate) use service::HandedService;
pub use supervisor::LIFELINE_ENV;

use std::net::IpAddr;
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use proto::process::{Hello, Join, Welcome, hello};
use proto::{Connection, Listener};

use claims::ClaimHolder;
use service::{Local, Stop};
use supervisor::Supervisor;

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
    /// The leader's `host:port` the members join at; the leader binds it
    /// for as long as it serves, so nodes may also join the running
    /// deployment. Required with several nodes.
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
    handoff: Arc<AtomicBool>,
    wake: UnixStream,
    thread: Option<JoinHandle<Served>>,
}

/// How the service thread ended: stopped (`None`), or stopped for a hot
/// restart with what it served and its descriptors.
type Served = anyhow::Result<Option<(HandedService, Vec<OwnedFd>)>>;

impl Daemon {
    /// Starts this node's daemon. The leader binds its process socket and
    /// its members' address and waits until every member below `nnodes`
    /// joined; a member joins the leader. Returns once the service runs.
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
        let handoff = Arc::new(AtomicBool::new(false));
        let stop = Stop {
            requested: Arc::clone(&requested),
            handoff: Arc::clone(&handoff),
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
            let (joins, members) = join_members(&options)?;
            let leader_address = leader_address(&options, &members);
            // While nodes may join, their processes reach this node's
            // Storage Server too: their Transfer Servers read its storage.
            let local = local(&options, leader_address, joins.is_some())?;
            let host = options.host.clone();
            let thread = std::thread::Builder::new()
                .name("daemon".into())
                .spawn(move || service::serve_leader(&host, local, listener, joins, members, &stop))
                .context("starting the daemon's service")?;
            (Some((address, options.socket.clone())), thread)
        } else {
            let leader = join_leader(&options)?;
            let address = leader.local_ip();
            let local = local(&options, address, true)?;
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
            handoff,
            wake,
            thread: Some(thread),
        })
    }

    /// Goes on with the service a predecessor handed over in a hot restart
    /// ([`Daemon::hand_over`]), on host `host`; `address` and `socket` are
    /// the leader's process socket, `None` on a member.
    ///
    /// # Errors
    ///
    /// Fails when the service thread cannot start.
    pub(crate) fn resume(
        host: String,
        address: Option<(String, PathBuf)>,
        handed: HandedService,
        descriptors: Vec<OwnedFd>,
    ) -> anyhow::Result<Self> {
        let (wake, wake_reader) = UnixStream::pair().context("creating the daemon's wake")?;
        wake_reader.set_nonblocking(true)?;
        let requested = Arc::new(AtomicBool::new(false));
        let handoff = Arc::new(AtomicBool::new(false));
        let stop = Stop {
            requested: Arc::clone(&requested),
            handoff: Arc::clone(&handoff),
            wake: wake_reader,
        };
        let thread = std::thread::Builder::new()
            .name("daemon".into())
            .spawn(move || service::resume(host, handed, descriptors, &stop))
            .context("starting the daemon's service")?;
        let (address, socket) = address.unzip();
        Ok(Self {
            address,
            socket,
            requested,
            handoff,
            wake,
            thread: Some(thread),
        })
    }

    /// Stops the service between two rounds and returns everything it
    /// serves, for a successor to go on with ([`Daemon::resume`]): its
    /// processes and connections outlive this daemon, and the process
    /// socket's file stays for the successor.
    ///
    /// # Errors
    ///
    /// Fails when the service ended or failed before it handed over.
    pub(crate) fn hand_over(&mut self) -> anyhow::Result<(HandedService, Vec<OwnedFd>)> {
        use std::io::Write as _;

        self.handoff.store(true, Ordering::Release);
        let _ = (&self.wake).write(&[1]);
        let thread = self
            .thread
            .take()
            .context("the daemon's service is not running")?;
        // The successor serves the socket from here on.
        self.socket = None;
        thread
            .join()
            .map_err(|_| anyhow::anyhow!("the daemon's service panicked"))??
            .context("the daemon's service stopped before it handed over")
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
        served?.map(drop)
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// This node's supervisor and claims. `remote_pages` says whether another
/// node's processes reach this node's Storage Server: a member node's
/// server serves the engine on the leader's page requests, and every node's
/// server of a deployment that other nodes may join exposes its storage to
/// their Transfer Servers' reads.
fn local(
    options: &DaemonOptions,
    address: Option<IpAddr>,
    remote_pages: bool,
) -> anyhow::Result<Local> {
    Ok(Local {
        supervisor: Supervisor::new(options.host.clone())
            .context("creating the daemon's lifeline")?,
        claims: ClaimHolder::new(options.executable.clone(), remote_pages, address),
        address,
    })
}

/// The leader's address its members join at, bound when the deployment
/// names one (`dist_init_addr`), and the members that joined before the
/// leader serves: every node rank below `nnodes` (other than the leader's)
/// within the join timeout. The address stays bound while the leader
/// serves, so a node may join the running deployment later (`Joins`).
fn join_members(
    options: &DaemonOptions,
) -> anyhow::Result<(Option<service::Joins>, Vec<service::Membership>)> {
    let Some(address) = options.dist_init_addr.as_deref() else {
        return Ok((None, Vec::new()));
    };
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

    let mut joins = service::Joins::new(listener);
    let deadline = Instant::now() + options.join_timeout;
    let mut members: Vec<service::Membership> = Vec::new();
    let awaited = |members: &[service::Membership]| {
        (1..options.nnodes)
            .filter(|rank| !members.iter().any(|member| member.node_rank == *rank))
            .count()
    };
    while awaited(&members) > 0 {
        anyhow::ensure!(
            Instant::now() < deadline,
            "{} of {} nodes joined within {:?}",
            options.nnodes as usize - awaited(&members),
            options.nnodes,
            options.join_timeout
        );
        let admitted = joins.admit(&members)?;
        members.extend(admitted);
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok((Some(joins), members))
}

/// The address the leader's members reach it at: the address a member
/// connected to, or the host `dist_init_addr` names when no member joined
/// yet.
fn leader_address(options: &DaemonOptions, members: &[service::Membership]) -> Option<IpAddr> {
    use std::net::ToSocketAddrs as _;

    members
        .first()
        .and_then(|member| member.connection.local_ip())
        .or_else(|| {
            options
                .dist_init_addr
                .as_deref()?
                .to_socket_addrs()
                .ok()?
                .map(|address| address.ip())
                .find(|address| !address.is_unspecified())
        })
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use proto::process::{
        Downstream, Engine, Request, Spawn, Stop, Upstream, downstream, reply, request, upstream,
    };

    use super::*;

    /// The next reply on `connection`.
    fn reply(connection: &mut Connection) -> proto::process::Reply {
        loop {
            let message: Downstream = connection
                .recv(Duration::from_secs(30))
                .unwrap()
                .expect("a reply in time");
            if let Some(downstream::Kind::Reply(reply)) = message.kind {
                return reply;
            }
        }
    }

    fn send(connection: &mut Connection, id: u64, kind: request::Kind) {
        connection
            .send(&Upstream {
                kind: Some(upstream::Kind::Request(Request {
                    id,
                    kind: Some(kind),
                })),
            })
            .unwrap();
    }

    /// A leader's service handed over goes on with what it served: a
    /// request on an engine connection the first service accepted is
    /// answered by the successor's, which ends the process the first one
    /// started.
    #[test]
    fn a_resumed_service_answers_the_connections_it_was_handed() {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("process.sock");
        let mut daemon = Daemon::start(DaemonOptions {
            nnodes: 1,
            node_rank: 0,
            host: "localhost".to_owned(),
            dist_init_addr: None,
            socket: socket.clone(),
            executable: PathBuf::from("uniserve"),
            join_timeout: Duration::from_secs(5),
        })
        .unwrap();
        let address = daemon.address().unwrap().to_owned();
        let mut engine = Connection::connect(&address).unwrap();
        engine
            .send(&Hello {
                major: proto::PROTOCOL_MAJOR,
                minor: proto::PROTOCOL_MINOR,
                build: proto::BUILD.to_owned(),
                peer: Some(hello::Peer::Engine(Engine {
                    deployment: "model".to_owned(),
                })),
            })
            .unwrap();
        let welcome: Welcome = engine.recv(Duration::from_secs(30)).unwrap().unwrap();
        assert!(welcome.refusal.is_empty(), "{}", welcome.refusal);
        send(
            &mut engine,
            1,
            request::Kind::Spawn(Spawn {
                host: "localhost".to_owned(),
                group: "model".to_owned(),
                rank: 0,
                generation: 1,
                executable: "sleep".to_owned(),
                arguments: vec!["600".to_owned()],
                ..Spawn::default()
            }),
        );
        let Some(reply::Kind::Spawned(spawned)) = reply(&mut engine).kind else {
            panic!("the spawn was not answered");
        };

        let (handed, descriptors) = daemon.hand_over().unwrap();
        // The handed-over daemon ends nothing and leaves its socket.
        drop(daemon);
        assert!(socket.exists());
        assert!(Path::new(&format!("/proc/{}", spawned.pid)).exists());
        let mut successor = Daemon::resume(
            "localhost".to_owned(),
            Some((address, socket.clone())),
            handed,
            descriptors,
        )
        .unwrap();

        send(
            &mut engine,
            2,
            request::Kind::Stop(Stop {
                group: "model".to_owned(),
                generation: None,
            }),
        );
        let stopped = reply(&mut engine);
        assert_eq!(stopped.id, 2);
        assert!(stopped.error.is_empty(), "{}", stopped.error);
        // The stopped process is this test's child: reap its zombie.
        let mut status = 0;
        // SAFETY: waits for this test's own child.
        unsafe { libc::waitpid(spawned.pid as libc::pid_t, &mut status, 0) };
        assert!(libc::WIFSIGNALED(status));
        successor.stop().unwrap();
    }
}
