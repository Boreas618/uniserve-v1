//! Supervision: what every node's daemon does for the processes of its
//! node.
//!
//! The daemons are peers. A daemon supervises the processes of its node
//! (`supervisor`): it starts them as its children, reaps them and reports
//! their exits. It keeps the deployment's Build Server and Storage Server on the
//! node serving with the deployment's settings, and holds the node's claim at
//! the Storage Server (`claims`), so a claim outlives every engine.
//!
//! The daemons talk the process interface (`proto::process`). Each member
//! daemon joins the leader at the leader's address; the leader then accepts
//! engines on its process socket, replies to their requests for its own node
//! and forwards the others to the member that owns the node (`service`).
//!
//! A hot restart hands a running daemon's service to its successor
//! ([`Daemon::hand_over`], [`Daemon::resume`]): the successor goes on with
//! its sockets, connections, processes and claims.

mod claims;
pub(crate) mod gpus;

pub(crate) mod processes;
mod service;
mod supervisor;

pub(crate) use service::{ServiceHandoff, readable};

/// The kernel's name of the executable image this process runs. The servers a daemon starts for
/// its deployment (the Build Server and the Storage Server) are commands of the running build, so the daemon
/// starts them from its own image: the path names that image even after the file it was loaded
/// from is replaced or removed.
pub(crate) const RUNNING_IMAGE: &str = "/proc/self/exe";

use std::net::IpAddr;
use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ::service::{Connection, Listener, Wake, Waker};
use anyhow::Context as _;
use proto::process::{Hello, Join, Welcome, hello};

use claims::ClaimHolder;
use service::{Local, Stop};
use supervisor::Supervisor;

/// How long the leader waits for every other node's daemon to join, and a
/// member to reach the leader: the nodes of a deployment are started
/// together, so this covers one that is starting rather than one waiting to
/// be scheduled.
pub const JOIN_TIMEOUT: Duration = Duration::from_secs(300);

/// A sender to the daemon's service thread that wakes it: the thread blocks on its sockets and its
/// processes' exits, and what another thread hands it through a channel reaches it with the wake
/// this sender fires after each message.
pub(crate) struct ToService<T> {
    sender: std::sync::mpsc::Sender<T>,
    wake: Wake,
}

impl<T> Clone for ToService<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            wake: self.wake.clone(),
        }
    }
}

impl<T> ToService<T> {
    /// Sends on `sender` and wakes with `wake`.
    pub(crate) fn new(sender: std::sync::mpsc::Sender<T>, wake: Wake) -> Self {
        Self { sender, wake }
    }

    /// Sends `value` and wakes the service thread.
    ///
    /// # Errors
    ///
    /// Gives `value` back when the service thread ended.
    pub(crate) fn send(&self, value: T) -> Result<(), std::sync::mpsc::SendError<T>> {
        self.sender.send(value)?;
        self.wake.wake();
        Ok(())
    }
}

/// Where a daemon runs and how it reaches the other nodes' daemons.
#[derive(Debug, Clone)]
pub struct DaemonOptions {
    /// The deployment, by its policy package's name: the daemon's Storage Server and Build
    /// Server on the node are the deployment's own.
    pub deployment: String,
    /// The node's runtime directory, the user's ([`::service::RuntimeDirectory::of_user`]):
    /// the deployment's Storage Server and Build Server keep their sockets in the deployment's
    /// directory there.
    pub runtime: ::service::RuntimeDirectory,
    /// The node's GPU registration file, where the daemon claims the GPUs its engines run on
    /// for the deployment ([`gpus::node`] on every node).
    pub gpus: PathBuf,
    /// Nodes of the deployment.
    pub nnodes: u32,
    /// This node's rank; the leader is rank 0.
    pub node_rank: u32,
    /// This node's host identity, as rank layouts name it.
    pub host: String,
    /// The leader's join endpoint: on the leader, the host its members reach it at; on a
    /// member, the endpoint the leader bound. Required with several nodes.
    pub dist_init_addr: Option<::service::Endpoint>,
    /// The leader's process socket, which engines connect to.
    pub socket: PathBuf,
    /// How long the leader waits for the members, and a member for the leader's welcome.
    pub join_timeout: Duration,
    /// The settings of the deployment's Build Server on this node, which the engines the node
    /// starts resolve their modules from (`build_config`).
    pub build_server: build_server::command::Settings,
    /// The settings of the deployment's Storage Server on every node it serves on
    /// (`Config::storage_settings`). The leader gives them to every claim of the
    /// deployment, so the Storage Server a node starts for it serves with them.
    pub storage: storage::Settings,
}

/// The daemon's downward path to the engines its service supervises, and the
/// reports they send up: the daemon commands an engine and reads its `Report`s
/// without holding its connection, which the service owns.
///
/// A clone of this shares the same ends. The reports have one consumer, the
/// updater's [`processes::Processes`], so that every acknowledgement and every
/// failure an engine states reaches the update in flight.
#[derive(Clone)]
pub struct ProcessCommands {
    /// Commands, by the engine that is to execute them.
    pub(crate) commands: ToService<(u64, proto::process::Command)>,
    /// Every report an engine sent, by the engine that sent it.
    pub(crate) reports:
        Arc<std::sync::Mutex<std::sync::mpsc::Receiver<(u64, proto::process::Report)>>>,
}

impl std::fmt::Debug for ProcessCommands {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ProcessCommands")
    }
}

impl ProcessCommands {
    /// Sends `command` to engine `engine`.
    ///
    /// # Errors
    ///
    /// Fails when the service stopped: nothing would carry the command.
    pub fn send(&self, engine: u64, command: proto::process::Command) -> anyhow::Result<()> {
        self.commands
            .send((engine, command))
            .map_err(|_| anyhow::anyhow!("the daemon's service stopped"))
    }

    /// The next report an engine sent, with the engine that sent it,
    /// waiting at most `timeout` for one.
    pub fn report(&self, timeout: Duration) -> Option<(u64, proto::process::Report)> {
        let deadline = Instant::now() + timeout;
        let reports = ::service::lock(&self.reports);
        // A report, the deadline, or a service that stopped sending each end
        // the wait.
        match reports.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(report) => Some(report),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => None,
        }
    }
}

/// A running daemon. Dropping it stops its service and ends every process
/// it supervises.
pub struct Daemon {
    /// The updater's ends of the service on the leader: its requests down,
    /// and the peers and exits up. `None` on a member. Taken once, by the
    /// updater's [`processes::Processes`]; behind a lock because a `Daemon` is
    /// shared between the HTTP threads and the channel ends are not
    /// themselves `Sync`.
    link: std::sync::Mutex<Option<processes::ProcessLink>>,
    /// The processes this node supervises, which the service keeps current:
    /// the updater and the deployment's status rows read them.
    table: Option<service::ProcessTable>,
    /// One token per thing the updater has to read, so the thread that
    /// advances the update waits instead of polling.
    awake: std::sync::Mutex<Option<std::sync::mpsc::Receiver<()>>>,
    /// The process socket engines connect to, on the leader. Its listener, which the service
    /// owns, removes a Unix socket's file when it closes.
    address: Option<String>,
    requested: Arc<AtomicBool>,
    handoff: Arc<AtomicBool>,
    /// The service's waker, fired with a stop.
    wake: Wake,
    thread: Option<JoinHandle<Served>>,
}

/// The ends of the leader's service: the service's own ([`service::Updater`]) and the
/// updater's ([`processes::ProcessLink`]), with the token channel that wakes the updater's thread
/// and the process table both read.
struct LeaderEnds {
    service: service::Updater,
    link: processes::ProcessLink,
    awake: std::sync::mpsc::Receiver<()>,
    table: service::ProcessTable,
}

impl LeaderEnds {
    /// The two ends of a leader's service whose waker fires `wake`.
    fn new(wake: &Wake) -> Self {
        let (commands, commands_rx) = std::sync::mpsc::channel();
        let (reports, reports_rx) = std::sync::mpsc::channel();
        let (requests, requests_rx) = std::sync::mpsc::channel();
        let (greetings, greetings_rx) = std::sync::mpsc::channel();
        let (exits, exits_rx) = std::sync::mpsc::channel();
        let (started, started_rx) = std::sync::mpsc::channel();
        let (activity, awake) = std::sync::mpsc::channel();
        let table: service::ProcessTable = Default::default();
        Self {
            service: service::Updater {
                commands: commands_rx,
                reports,
                requests: requests_rx,
                greetings,
                exits,
                started,
                wake: activity,
                processes: Arc::clone(&table),
            },
            link: processes::ProcessLink {
                requests: ToService::new(requests, wake.clone()),
                greetings: greetings_rx,
                exits: exits_rx,
                started: started_rx,
                commands: ProcessCommands {
                    commands: ToService::new(commands, wake.clone()),
                    reports: Arc::new(std::sync::Mutex::new(reports_rx)),
                },
            },
            awake,
            table,
        }
    }
}

/// How the service thread ended: stopped (`None`), or stopped for a hot
/// restart with what it served and its descriptors.
type Served = anyhow::Result<Option<(ServiceHandoff, Vec<OwnedFd>)>>;

impl Daemon {
    /// Binds the leader's join listener for the endpoint `dist_init_addr` states: its port on
    /// every interface (port 0 picks one). Returns the listener and the endpoint members join at,
    /// the stated host with the bound port, which the leader's readiness record states.
    ///
    /// # Errors
    ///
    /// Fails when the endpoint is no TCP endpoint or its port cannot be bound.
    pub fn bind_join(
        dist_init_addr: &::service::Endpoint,
    ) -> anyhow::Result<(Listener, ::service::Endpoint)> {
        let ::service::Endpoint::Tcp { host, port } = dist_init_addr else {
            anyhow::bail!("the leader's join endpoint {dist_init_addr} is no TCP endpoint");
        };
        let listener = Listener::bind(&::service::Endpoint::Tcp {
            host: std::net::Ipv4Addr::UNSPECIFIED.to_string(),
            port: *port,
        })
        .with_context(|| format!("binding the leader's join endpoint {dist_init_addr}"))?;
        let bound = listener
            .endpoint()
            .port()
            .context("a TCP listener has a port")?;
        Ok((
            listener,
            ::service::Endpoint::Tcp {
                host: host.clone(),
                port: bound,
            },
        ))
    }

    /// Starts this node's daemon. The leader binds its process socket and, given the join
    /// listener its caller bound (`join`, [`Daemon::bind_join`]), waits until every member below
    /// `nnodes` joined there; a member joins the leader. Returns once the service runs.
    ///
    /// # Errors
    ///
    /// Fails when a socket cannot be bound, a member does not join in time,
    /// or the leader cannot be reached or refuses this member.
    pub fn start(options: DaemonOptions, join: Option<Listener>) -> anyhow::Result<Self> {
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
        let waker = Waker::new().context("creating the daemon's waker")?;
        let wake = waker.wake();
        let requested = Arc::new(AtomicBool::new(false));
        let handoff = Arc::new(AtomicBool::new(false));
        let mut stop = Stop {
            requested: Arc::clone(&requested),
            handoff: Arc::clone(&handoff),
            waker,
        };

        let (address, thread, ends) = if options.node_rank == 0 {
            let (joins, members) = join_members(&options, join)?;
            let leader_address = leader_address(&options)?;
            // Engines may run on any joined node. A deployment with a
            // network address exposes process control on that interface;
            // single-host deployments retain their Unix socket.
            let endpoint = match leader_address {
                Some(ip) => ::service::Endpoint::tcp(std::net::SocketAddr::new(ip, 0)),
                None => ::service::Endpoint::unix(&options.socket),
            };
            let listener = Listener::bind(&endpoint)
                .with_context(|| format!("binding the process socket {endpoint}"))?;
            let address = listener.endpoint().to_string();
            // While nodes may join, their processes reach this node's
            // Storage Server too: their Transfer Servers read its storage.
            let local = local(&options, leader_address, joins.is_some(), wake.clone())?;
            let host = options.host.clone();
            let LeaderEnds {
                service,
                link,
                awake,
                table,
            } = LeaderEnds::new(&wake);
            let thread = std::thread::Builder::new()
                .name("daemon".into())
                .spawn(move || {
                    service::serve_leader(
                        &host, local, listener, joins, members, service, &mut stop,
                    )
                })
                .context("starting the daemon's service")?;
            (Some(address), thread, Some((link, awake, table)))
        } else {
            let leader = join_leader(&options)?;
            let address = leader.local_addr().map(|address| address.ip());
            let local = local(&options, address, true, wake.clone())?;
            let thread = std::thread::Builder::new()
                .name("daemon".into())
                .spawn(move || service::serve_member(local, leader, &mut stop))
                .context("starting the daemon's service")?;
            (None, thread, None)
        };
        Ok(Self::running(
            address, requested, handoff, wake, thread, ends,
        ))
    }

    /// A running daemon of the service thread `thread`, which `wake` wakes; `ends` are the
    /// updater's ends of a leader's service, `None` on a member.
    fn running(
        address: Option<String>,
        requested: Arc<AtomicBool>,
        handoff: Arc<AtomicBool>,
        wake: Wake,
        thread: JoinHandle<Served>,
        ends: Option<(
            processes::ProcessLink,
            std::sync::mpsc::Receiver<()>,
            service::ProcessTable,
        )>,
    ) -> Self {
        let (link, awake, table) = match ends {
            Some((link, awake, table)) => (Some(link), Some(awake), Some(table)),
            None => (None, None, None),
        };
        Self {
            table,
            link: std::sync::Mutex::new(link),
            awake: std::sync::Mutex::new(awake),
            address,
            requested,
            handoff,
            wake,
            thread: Some(thread),
        }
    }

    /// Goes on with the service a predecessor handed over in a hot restart
    /// ([`Daemon::hand_over`]), on host `host`; `address` is the leader's process socket, `None`
    /// on a member.
    ///
    /// # Errors
    ///
    /// Fails when the service thread cannot start.
    pub(crate) fn resume(
        host: String,
        address: Option<String>,
        service: ServiceHandoff,
        descriptors: Vec<OwnedFd>,
    ) -> anyhow::Result<Self> {
        let waker = Waker::new().context("creating the daemon's waker")?;
        let wake = waker.wake();
        let requested = Arc::new(AtomicBool::new(false));
        let handoff = Arc::new(AtomicBool::new(false));
        let mut stop = Stop {
            requested: Arc::clone(&requested),
            handoff: Arc::clone(&handoff),
            waker,
        };
        let leader = matches!(service, ServiceHandoff::Leader { .. });
        let LeaderEnds {
            service: ends,
            link,
            awake,
            table,
        } = LeaderEnds::new(&wake);
        let thread = std::thread::Builder::new()
            .name("daemon".into())
            .spawn(move || service::resume(host, service, descriptors, ends, &mut stop))
            .context("starting the daemon's service")?;
        Ok(Self::running(
            address,
            requested,
            handoff,
            wake,
            thread,
            leader.then_some((link, awake, table)),
        ))
    }

    /// Stops the service between two rounds and returns everything it
    /// serves, for a successor to go on with ([`Daemon::resume`]): its
    /// processes and connections outlive this daemon, and the process
    /// socket's file stays for the successor.
    ///
    /// # Errors
    ///
    /// Fails when the service ended or failed before it handed over.
    pub(crate) fn hand_over(&mut self) -> anyhow::Result<(ServiceHandoff, Vec<OwnedFd>)> {
        self.handoff.store(true, Ordering::Release);
        self.wake.wake();
        let thread = self
            .thread
            .take()
            .context("the daemon's service is not running")?;
        thread
            .join()
            .map_err(|_| anyhow::anyhow!("the daemon's service panicked"))??
            .context("the daemon's service stopped before it handed over")
    }

    /// The updater's ends of this node's service: `None` on a member daemon,
    /// and once taken, `None` again: one owner reads them.
    pub(crate) fn process_link(&self) -> Option<processes::ProcessLink> {
        ::service::lock(&self.link).take()
    }

    /// The updater of `deployment` over this node's service and new epoch records, with the
    /// tokens that wake the thread advancing it. `None` on a member daemon, which commands no
    /// deployment's engines. The records (`Updater::epochs`) report the node's processes from
    /// the table the service writes as they start and end; the updater's `Processes` adds what
    /// each launch is, and writes each process's start, report and exit through `recorder`.
    pub(crate) fn updater(
        &self,
        deployment: String,
        recorder: crate::observation::record::Recorder,
    ) -> Option<(Arc<crate::updates::Updater>, std::sync::mpsc::Receiver<()>)> {
        let link = self.process_link()?;
        let awake = ::service::lock(&self.awake).take()?;
        let epochs = Arc::new(crate::epoch::Epochs::new(self.table.clone()?));
        let processes = processes::Processes::new(Arc::clone(&epochs), deployment, link, recorder);
        Some((
            Arc::new(crate::updates::Updater::new(epochs, processes)),
            awake,
        ))
    }

    /// Advances the update in flight whenever the service reports something,
    /// until the service ends: a phase ends on acknowledgements, so the
    /// thread that advances it is woken by the reports, never by a clock.
    pub(crate) fn advance_updates(
        updater: Arc<crate::updates::Updater>,
        awake: std::sync::mpsc::Receiver<()>,
    ) -> anyhow::Result<()> {
        std::thread::Builder::new()
            .name("updates".into())
            .spawn(move || {
                while awake.recv().is_ok() {
                    updater.advance();
                }
            })
            .context("starting the updater's thread")?;
        Ok(())
    }

    /// The leader's process socket, which engines connect to; `None` on a
    /// member.
    pub fn address(&self) -> Option<&str> {
        self.address.as_deref()
    }

    /// What stops this daemon's service from another thread while one waits for its end
    /// ([`Daemon::wait`]): the stop [`Daemon::stop`] asks for, without the join.
    pub fn stopper(&self) -> Stopper {
        Stopper {
            requested: Arc::clone(&self.requested),
            wake: self.wake.clone(),
        }
    }

    /// Waits until the service ends: on a member, when the leader's
    /// connection closes; on any node, once a [`Stopper`] stopped it.
    ///
    /// # Errors
    ///
    /// Returns the service's failure.
    pub fn wait(mut self) -> anyhow::Result<()> {
        self.join()
    }

    /// Stops the service and ends every process it supervises.
    pub fn stop(&mut self) -> anyhow::Result<()> {
        self.requested.store(true, Ordering::Release);
        self.wake.wake();
        self.join()
    }

    fn join(&mut self) -> anyhow::Result<()> {
        let Some(thread) = self.thread.take() else {
            return Ok(());
        };
        thread
            .join()
            .map_err(|_| anyhow::anyhow!("the daemon's service panicked"))?
            .map(drop)
    }
}

/// Stops a daemon's service from any thread ([`Daemon::stopper`]).
#[derive(Clone)]
pub struct Stopper {
    requested: Arc<AtomicBool>,
    wake: Wake,
}

impl Stopper {
    /// Asks the service to stop: it ends every process it supervises and its thread ends, which
    /// ends the [`Daemon::wait`] of whoever waits for it.
    pub fn stop(&self) {
        self.requested.store(true, Ordering::Release);
        self.wake.wake();
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
    wake: Wake,
) -> anyhow::Result<Local> {
    Ok(Local::new(
        Supervisor::new(
            options.host.clone(),
            options.runtime.clone(),
            options.deployment.clone(),
            gpus::Gpus::open(&options.gpus, &options.deployment).with_context(|| {
                format!(
                    "opening the GPU registration file {}",
                    options.gpus.display()
                )
            })?,
        )
        .context("creating the daemon's lifeline")?
        .with_build_server(options.build_server.clone()),
        ClaimHolder::new(
            options.runtime.clone(),
            options.deployment.clone(),
            remote_pages,
            address,
            options.storage.clone(),
        ),
        address,
        wake,
    ))
}

/// The members that joined before the leader serves, at the join listener the caller bound
/// (`join`, [`Daemon::bind_join`]) when the deployment names a join endpoint
/// (`dist_init_addr`): every node rank below `nnodes` (other than the leader's) within the join
/// timeout. The wait blocks on the listener and the joining connections, so a join ends it as it
/// arrives. The listener stays bound while the leader serves, so a node may join the running
/// deployment later (`Joins`).
fn join_members(
    options: &DaemonOptions,
    join: Option<Listener>,
) -> anyhow::Result<(Option<service::Joins>, Vec<service::Membership>)> {
    let Some(listener) = join else {
        anyhow::ensure!(
            options.nnodes == 1,
            "a leader of {} nodes needs its join listener",
            options.nnodes
        );
        return Ok((None, Vec::new()));
    };
    tracing::info!(endpoint = %listener.endpoint(), nodes = options.nnodes, "awaiting the deployment's other nodes");

    let mut joins = service::Joins::new(listener);
    let deadline = Instant::now() + options.join_timeout;
    let mut members: Vec<service::Membership> = Vec::new();
    let awaited = |members: &[service::Membership]| {
        (1..options.nnodes)
            .filter(|rank| !members.iter().any(|member| member.node_rank == *rank))
            .count()
    };
    loop {
        let admitted = joins.admit(&members)?;
        members.extend(admitted);
        if awaited(&members) == 0 {
            return Ok((Some(joins), members));
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        anyhow::ensure!(
            !remaining.is_zero(),
            "{} of {} nodes joined within {:?}",
            options.nnodes as usize - awaited(&members),
            options.nnodes,
            options.join_timeout
        );
        let mut descriptors: Vec<libc::pollfd> = joins.poll_fds().collect();
        ::service::wait_ready(&mut descriptors, Some(remaining))?;
    }
}

/// The address the leader's members reach it at: the host of the join endpoint the deployment
/// declares (`dist_init_addr`), which the members dial, as the system resolver resolves it.
/// `None` for a deployment that names no join endpoint, whose processes all run on this host.
///
/// # Errors
///
/// Fails when the declared host resolves to no address a member can reach.
fn leader_address(options: &DaemonOptions) -> anyhow::Result<Option<IpAddr>> {
    use std::net::ToSocketAddrs as _;

    let Some(endpoint) = &options.dist_init_addr else {
        return Ok(None);
    };
    let ::service::Endpoint::Tcp { host, port } = endpoint else {
        anyhow::bail!("the leader's join endpoint {endpoint} is no TCP endpoint");
    };
    let address = (host.as_str(), *port)
        .to_socket_addrs()
        .with_context(|| format!("resolving the leader's join host {host}"))?
        .map(|address| address.ip())
        .find(|address| !address.is_unspecified())
        .with_context(|| {
            format!("the leader's join host {host} names no address another node can reach")
        })?;
    Ok(Some(address))
}

/// A member's join: connects to the leader at the join endpoint the leader reported bound
/// (`dist_init_addr`, from the leader's readiness record), states this node and the address its
/// connection reaches the leader from, and reads the leader's welcome.
fn join_leader(options: &DaemonOptions) -> anyhow::Result<Connection> {
    let address = options
        .dist_init_addr
        .as_ref()
        .context("a member needs the leader's address")?;
    // The leader binds its join listener before it starts anything slow and reports it, so a
    // member started from that report connects at once; a refused connection is a leader that
    // is not serving that endpoint.
    let mut connection =
        Connection::connect(address).with_context(|| format!("joining the leader at {address}"))?;
    let reached_from = connection
        .local_addr()
        .context("a member reaches the leader over TCP")?
        .ip();
    connection.deliver(&Hello {
        major: proto::ENGINE_INTERFACE_VERSION,
        build: proto::BUILD.to_owned(),
        peer: Some(hello::Peer::Join(Join {
            node_rank: options.node_rank,
            host: options.host.clone(),
            address: reached_from.to_string(),
        })),
    })?;
    let welcome: Welcome = connection
        .recv(options.join_timeout)?
        .context("the leader did not reply to this node's hello")?;
    if let Some(refusal) = welcome.refusal {
        return Err(anyhow::Error::new(refusal).context("the leader refused this node"));
    }
    tracing::info!(leader = %address, node_rank = options.node_rank, "joined the leader");
    Ok(connection)
}

#[cfg(test)]
mod tests {
    use proto::process::{
        Downstream, Engine, NodesQuery, Request, Reserve, Stop, Upstream, downstream, reply,
        request, upstream,
    };

    use super::*;

    /// Sends request `id` of `kind` on `connection` and returns its reply.
    fn ask(connection: &mut Connection, id: u64, kind: request::Kind) -> proto::process::Reply {
        connection
            .deliver(&Upstream {
                kind: Some(upstream::Kind::Request(Request {
                    id,
                    kind: Some(kind),
                })),
            })
            .unwrap();
        loop {
            let message: Downstream = connection.recv_next().unwrap();
            if let Some(downstream::Kind::Reply(reply)) = message.kind {
                assert_eq!(reply.id, id);
                return reply;
            }
        }
    }

    /// A leader's service handed over goes on with what it served: the successor answers a
    /// request on a connection the first service accepted, holds the socket the first one
    /// reserved, and ends it with the group's stop.
    #[test]
    fn a_resumed_service_serves_the_connections_and_reservations_of_its_handoff() {
        for network in [false, true] {
            resumed_service(network);
        }
    }

    fn resumed_service(network: bool) {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("process.sock");
        let join = network.then(|| Daemon::bind_join(&"127.0.0.1:0".parse().unwrap()).unwrap());
        let mut daemon = Daemon::start(
            DaemonOptions {
                deployment: "model".to_owned(),
                runtime: ::service::RuntimeDirectory::within(directory.path()).unwrap(),
                gpus: directory.path().join(gpus::NODE),
                nnodes: 1,
                node_rank: 0,
                host: "localhost".to_owned(),
                dist_init_addr: join.as_ref().map(|(_, endpoint)| endpoint.clone()),
                socket: socket.clone(),
                join_timeout: JOIN_TIMEOUT,
                build_server: build_server::command::Settings::default(),
                storage: storage::Settings::default(),
            },
            join.map(|(listener, _)| listener),
        )
        .unwrap();
        let address = daemon.address().unwrap().to_owned();
        let mut client = Connection::connect(&address.parse().unwrap()).unwrap();
        client
            .deliver(&Hello {
                major: proto::ENGINE_INTERFACE_VERSION,
                build: proto::BUILD.to_owned(),
                // A client of the socket: no engine number, no interface.
                peer: Some(hello::Peer::Engine(Engine {
                    deployment: "model".to_owned(),
                    engine: 0,
                    interface: 0,
                })),
            })
            .unwrap();
        let welcome: Welcome = client.recv_next().unwrap();
        assert_eq!(welcome.refusal, None);
        let reserved = ask(
            &mut client,
            1,
            request::Kind::Reserve(Reserve {
                host: "localhost".to_owned(),
                group: "model".to_owned(),
                engine: None,
                routable: false,
            }),
        );
        let Some(reply::Kind::Reserved(reserved)) = reserved.kind else {
            panic!("the reservation got no socket: {reserved:?}");
        };

        let (handoff, descriptors) = daemon.hand_over().unwrap();
        // The daemon that handed over ends nothing and leaves its socket.
        drop(daemon);
        assert_eq!(socket.exists(), !network);
        let mut successor =
            Daemon::resume("localhost".to_owned(), Some(address), handoff, descriptors).unwrap();

        let nodes = ask(&mut client, 2, request::Kind::Nodes(NodesQuery {}));
        assert!(
            matches!(nodes.kind, Some(reply::Kind::Nodes(_))),
            "{nodes:?}"
        );
        std::net::TcpStream::connect(&reserved.address)
            .expect("the successor holds the reserved socket");
        let stopped = ask(
            &mut client,
            3,
            request::Kind::Stop(Stop {
                group: "model".to_owned(),
                generation: None,
            }),
        );
        assert!(stopped.error.is_empty(), "{}", stopped.error);
        // The group's stop closed the reservation the successor adopted: a launch that inherits
        // it is refused before anything starts.
        let refused = ask(
            &mut client,
            4,
            request::Kind::Spawn(proto::process::Spawn {
                host: "localhost".to_owned(),
                group: "model".to_owned(),
                inherit: vec![reserved.reservation],
                ..proto::process::Spawn::default()
            }),
        );
        assert!(refused.error.contains("no reservation"), "{refused:?}");
        successor.stop().unwrap();
    }
}
