//! The head's addresses: where its ranks and launchers connect and register,
//! and where a worker group's ranks rendezvous.
//!
//! The head listens on one Unix socket path and one TCP port for the whole
//! deployment. A rank connects to the head when it starts, through the Unix
//! socket on the head's host and through TCP from any other host, and its
//! first frame is its `Registration`: worker group, rank, launch generation
//! and build version. [`Head::collect`] gathers the registrations of one
//! launch and hands each rank's connection to that launch, which answers with
//! the rank's launch descriptor. The connection then serves the rank for its
//! lifetime, and its close is the rank's death.
//!
//! The launcher of another host connects to the same TCP port and registers
//! its host identity; [`Head::take_launchers`] hands its connection to the
//! launcher registry (`launcher`).
//!
//! A host that runs ranks therefore accepts no inbound connection.

use std::collections::{BTreeMap, HashMap};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use uniserve_ipc::{Channel, Listener, Registration, VERSION};

/// Deadline for every rank of one launch to register, measured from the
/// start of `Head::collect`, which runs after every rank of the group has
/// been launched.
///
/// A rank registers before it builds its model, so this covers process start
/// and imports rather than weight loading.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(300);

/// Interval between accept attempts while no rank has connected.
const ACCEPT_INTERVAL: Duration = Duration::from_millis(20);

/// Deadline for one accepted connection to deliver its registration.
const REGISTRATION_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// A worker group's collective rendezvous.
pub(crate) struct Rendezvous {
    /// The store address every rank connects to, as `host:port`.
    pub address: String,
    /// The bound, listening socket the group's first rank serves the store
    /// on, held by the process that spawns that rank until it hands the
    /// socket over. `None` when the first rank runs on another host, whose
    /// launcher holds that host's reservation, and once the socket has been
    /// taken for the spawn.
    pub listener: Option<TcpListener>,
    /// The descriptor number the group's first rank inherits the socket at
    /// when a launcher on another host holds it, as that launcher reported.
    pub listen_fd: Option<std::os::fd::RawFd>,
}

/// Reserves the address a worker group's ranks rendezvous at.
///
/// The collective store is a TCP store so ranks on different hosts can reach
/// it; an instance on one host reserves a loopback address, and one spanning
/// hosts reserves an address those hosts can route to (see `interface`).
/// The head calls this when the group's first rank runs on its own host: it
/// binds the socket here, and that rank, which it spawns, inherits the socket
/// and serves the store on it. The port therefore stays bound from this
/// reservation until that rank exits, so no other process on the host can
/// take it between the reservation and the rank's start.
pub(crate) fn reserve_rendezvous(head: Option<IpAddr>) -> anyhow::Result<Rendezvous> {
    let (bind, advertise) = interface(head);
    let listener = TcpListener::bind(SocketAddr::from((bind, 0)))
        .context("reserving the collective rendezvous address")?;
    let port = listener
        .local_addr()
        .context("reading the collective rendezvous address")?
        .port();
    Ok(Rendezvous {
        address: format!("{advertise}:{port}"),
        listener: Some(listener),
        listen_fd: None,
    })
}

/// Returns the interface to bind and the host a descriptor names.
///
/// `head` is an address of this host that the other hosts route to, and none
/// when every rank runs here. A loopback address reaches only the host that
/// binds it, so an instance spanning hosts binds every interface and names
/// itself by that routable address rather than by the wildcard, which names
/// no interface, or by the placement's host identity, which need not resolve
/// on other hosts.
fn interface(head: Option<IpAddr>) -> (Ipv4Addr, String) {
    match head {
        Some(address) => (Ipv4Addr::UNSPECIFIED, address.to_string()),
        None => (Ipv4Addr::LOCALHOST, Ipv4Addr::LOCALHOST.to_string()),
    }
}

/// The deployment's head, shared by its worker groups.
pub(crate) type SharedHead = Arc<Mutex<Head>>;

/// Locks the head, reporting a poisoned lock as an error.
pub(crate) fn lock(head: &Mutex<Head>) -> anyhow::Result<std::sync::MutexGuard<'_, Head>> {
    head.lock()
        .map_err(|_| anyhow::anyhow!("the head's listener is poisoned"))
}

/// A rank that has registered and that no launch has taken yet.
struct RegisteredRank {
    worker_id: String,
    rank: u32,
    generation: u64,
    channel: Channel,
}

/// The address ranks and launchers connect to, and those that have
/// registered there.
pub(crate) struct Head {
    listener: Listener,
    /// Holds the Unix socket's path; dropping it removes the directory.
    _directory: tempfile::TempDir,
    /// The generation of each worker group's latest launch.
    generations: HashMap<String, u64>,
    /// Accepted connections whose registration has not arrived, with the
    /// time each was accepted.
    connecting: Vec<(Channel, Instant)>,
    /// Registered ranks no launch has taken yet.
    ranks: Vec<RegisteredRank>,
    /// Registered launchers the registry has not taken yet, by host.
    launchers: Vec<(String, Channel)>,
    /// Registrations refused for stating another build, kept so whoever
    /// waits for such a rank or launcher fails by name.
    refused: Vec<Registration>,
}

impl Head {
    /// Binds the Unix socket path and the TCP port ranks and launchers
    /// connect to.
    ///
    /// `routable` says whether some rank runs on another host: the TCP port
    /// then binds every interface, and loopback otherwise, since a rank on
    /// another host cannot reach a loopback address and a deployment on one
    /// host does not need a routable one.
    pub(crate) fn bind(routable: bool) -> anyhow::Result<SharedHead> {
        let directory = tempfile::Builder::new()
            .prefix("uniserve-head")
            .tempdir()
            .context("creating the directory of the head's socket")?;
        let unix = format!("unix:{}", directory.path().join("head.sock").display());
        let interface = if routable {
            Ipv4Addr::UNSPECIFIED
        } else {
            Ipv4Addr::LOCALHOST
        };
        let tcp = SocketAddr::from((interface, 0)).to_string();
        let listener =
            Listener::bind(&[&unix, &tcp]).context("binding the head's rank addresses")?;
        Ok(Arc::new(Mutex::new(Self {
            listener,
            _directory: directory,
            generations: HashMap::new(),
            connecting: Vec::new(),
            ranks: Vec::new(),
            launchers: Vec::new(),
            refused: Vec::new(),
        })))
    }

    /// Returns the address a rank connects to.
    ///
    /// A rank on the head's host takes the Unix socket. A rank elsewhere
    /// takes the TCP port at `reachable`, an address of this host that the
    /// other hosts route to.
    pub(crate) fn address(&self, reachable: Option<IpAddr>) -> anyhow::Result<String> {
        let addresses = self.listener.addresses();
        match reachable {
            None => Ok(addresses[0].clone()),
            Some(host) => {
                let port = addresses[1]
                    .parse::<SocketAddr>()
                    .context("reading the head's TCP port")?
                    .port();
                Ok(SocketAddr::new(host, port).to_string())
            }
        }
    }

    /// Starts a launch of `worker_id` and returns its generation, which
    /// every rank of the launch states when it registers.
    pub(crate) fn begin_launch(&mut self, worker_id: &str) -> u64 {
        let generation = self.generations.entry(worker_id.to_owned()).or_default();
        *generation += 1;
        *generation
    }

    /// Collects the connection of every rank of one launch, ordered by rank.
    ///
    /// `alive` is consulted whenever the launch is still incomplete, so a
    /// rank that dies before registering fails the launch by name instead of
    /// consuming the deadline.
    ///
    /// Fails on an error from `alive`, when the deadline passes with ranks
    /// unregistered (named in the error), when a rank of this launch states
    /// another build than the head's, and when a registration names a rank
    /// outside `0..expected` or one that already registered. Connections of
    /// other worker groups are kept for their own launches; a rank of an
    /// earlier generation of this group is refused by closing its connection.
    pub(crate) fn collect(
        &mut self,
        worker_id: &str,
        generation: u64,
        expected: usize,
        mut alive: impl FnMut() -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<Channel>> {
        let deadline = Instant::now() + REGISTRATION_TIMEOUT;
        let mut ranks = BTreeMap::<usize, Channel>::new();
        loop {
            self.accept()?;
            self.read_registrations();

            for refused in &self.refused {
                if let Registration::Rank {
                    worker_id: refused_worker,
                    rank,
                    generation: refused_generation,
                    version,
                } = refused
                    && refused_worker == worker_id
                    && *refused_generation == generation
                {
                    anyhow::bail!(
                        "rank {rank} of worker {worker_id} is build {version} and the head is \
                         build {VERSION}; reinstall so that both run the same build"
                    );
                }
            }

            // Take this launch's registrations and leave the other groups'.
            let mut index = 0;
            while index < self.ranks.len() {
                if self.ranks[index].worker_id != worker_id {
                    index += 1;
                    continue;
                }
                let registered = self.ranks.swap_remove(index);
                if registered.generation != generation {
                    // Dropping the connection is the refusal: the rank exits
                    // when its connection closes.
                    tracing::warn!(
                        worker = worker_id,
                        rank = registered.rank,
                        generation = registered.generation,
                        "refusing a rank of a launch that has ended"
                    );
                    continue;
                }
                let rank = registered.rank as usize;
                anyhow::ensure!(
                    rank < expected,
                    "rank {rank} registered for worker {worker_id}, which has {expected} ranks"
                );
                anyhow::ensure!(
                    ranks.insert(rank, registered.channel).is_none(),
                    "rank {rank} of worker {worker_id} registered twice"
                );
            }
            if ranks.len() == expected {
                return Ok(ranks.into_values().collect());
            }

            alive()?;
            if Instant::now() >= deadline {
                let missing: Vec<usize> = (0..expected)
                    .filter(|rank| !ranks.contains_key(rank))
                    .collect();
                anyhow::bail!(
                    "ranks {missing:?} of worker {worker_id} did not register within {}s",
                    REGISTRATION_TIMEOUT.as_secs()
                );
            }
            self.park()?;
        }
    }

    /// Returns the TCP address the head listens at, as it was bound.
    pub(crate) fn tcp_address(&self) -> &str {
        &self.listener.addresses()[1]
    }

    /// Takes the connection of every launcher that has registered since the
    /// last call, with the host identity each registered.
    ///
    /// Fails when a launcher of one of `hosts` stated another build than the
    /// head's.
    pub(crate) fn take_launchers(
        &mut self,
        hosts: &[String],
    ) -> anyhow::Result<Vec<(String, Channel)>> {
        self.accept()?;
        self.read_registrations();
        for refused in &self.refused {
            if let Registration::Launcher { host, version } = refused
                && hosts.contains(host)
            {
                anyhow::bail!(
                    "the launcher of host {host} is build {version} and the head is build \
                     {VERSION}; reinstall so that both run the same build"
                );
            }
        }
        Ok(std::mem::take(&mut self.launchers))
    }

    /// Waits a short interval for a new connection or a registration.
    pub(crate) fn park(&self) -> anyhow::Result<()> {
        let mut descriptors = self.listener.poll_fds();
        for (channel, _) in &self.connecting {
            descriptors.extend(channel.poll_fds());
        }
        super::park_descriptors(&descriptors, ACCEPT_INTERVAL)
    }

    /// Takes every waiting connection.
    fn accept(&mut self) -> anyhow::Result<()> {
        while let Some(channel) = self
            .listener
            .try_accept()
            .context("accepting a rank's connection")?
        {
            self.connecting.push((channel, Instant::now()));
        }
        Ok(())
    }

    /// Reads the registration of each connection that has sent one.
    ///
    /// A connection that fails, sends something else, or sends nothing within
    /// `REGISTRATION_READ_TIMEOUT` is dropped, which closes it. One that
    /// states another build than this one is refused the same way and
    /// remembered in `refused`.
    fn read_registrations(&mut self) {
        let mut index = 0;
        while index < self.connecting.len() {
            let (channel, accepted) = &mut self.connecting[index];
            match channel.try_recv::<Registration>() {
                Ok(None) if accepted.elapsed() < REGISTRATION_READ_TIMEOUT => {
                    index += 1;
                    continue;
                }
                Ok(None) => tracing::warn!("a connection to the head sent no registration"),
                Ok(Some(registration)) if registration.version() != VERSION => {
                    tracing::error!(
                        ?registration,
                        head = VERSION,
                        "refusing a registration of another build"
                    );
                    self.refused.push(registration);
                }
                Ok(Some(registration)) => {
                    let (channel, _) = self.connecting.swap_remove(index);
                    match registration {
                        Registration::Rank {
                            worker_id,
                            rank,
                            generation,
                            ..
                        } => self.ranks.push(RegisteredRank {
                            worker_id,
                            rank,
                            generation,
                            channel,
                        }),
                        Registration::Launcher { host, .. } => self.launchers.push((host, channel)),
                    }
                    continue;
                }
                Err(error) => {
                    tracing::warn!(%error, "a connection to the head sent no valid registration");
                }
            }
            self.connecting.swap_remove(index);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Connects to the head and registers as one rank.
    fn register(address: &str, worker: &str, rank: u32, generation: u64, version: &str) -> Channel {
        let mut channel = Channel::connect(address, Duration::from_secs(5)).unwrap();
        channel
            .send(&Registration::Rank {
                worker_id: worker.into(),
                rank,
                generation,
                version: version.into(),
            })
            .unwrap();
        channel
    }

    /// Waits until the head closes `channel`.
    fn closed(channel: &mut Channel) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if channel.try_recv::<Registration>().is_err() {
                return true;
            }
            channel.wait(Duration::from_millis(20)).unwrap();
        }
        false
    }

    #[test]
    fn a_launch_collects_its_own_ranks_over_both_addresses_in_rank_order() {
        let head = Head::bind(false).unwrap();
        let mut head = lock(&head).unwrap();
        let unix = head.address(None).unwrap();
        let tcp = head.address(Some(Ipv4Addr::LOCALHOST.into())).unwrap();
        assert!(unix.starts_with("unix:") && !tcp.starts_with("unix:"));

        let text = head.begin_launch("text");
        let image = head.begin_launch("image");
        let _other = register(&unix, "image", 0, image, VERSION);
        let mut second = register(&tcp, "text", 1, text, VERSION);
        let mut first = register(&unix, "text", 0, text, VERSION);

        // Each collected connection answers the rank that registered on it.
        let mut ranks = head.collect("text", text, 2, || Ok(())).unwrap();
        assert_eq!(ranks.len(), 2);
        for (rank, channel) in ranks.iter_mut().enumerate() {
            channel
                .send(&uniserve_ipc::WorkerRequest::Info {
                    message_id: Some(rank as u64),
                })
                .unwrap();
        }
        for (rank, peer) in [&mut first, &mut second].into_iter().enumerate() {
            let deadline = Instant::now() + Duration::from_secs(5);
            let request = loop {
                if let Some(request) = peer.try_recv::<uniserve_ipc::WorkerRequest>().unwrap() {
                    break request;
                }
                assert!(Instant::now() < deadline);
                peer.wait(Duration::from_millis(20)).unwrap();
            };
            assert_eq!(request.message_id(), Some(rank as u64));
        }

        // The other group's rank stayed for its own launch.
        assert_eq!(head.collect("image", image, 1, || Ok(())).unwrap().len(), 1);
    }

    #[test]
    fn a_rank_of_an_ended_launch_is_refused_and_its_connection_closed() {
        let head = Head::bind(false).unwrap();
        let mut head = lock(&head).unwrap();
        let address = head.address(None).unwrap();

        let ended = head.begin_launch("text");
        let current = head.begin_launch("text");
        let mut stale = register(&address, "text", 0, ended, VERSION);
        let _current = register(&address, "text", 0, current, VERSION);

        assert_eq!(
            head.collect("text", current, 1, || Ok(())).unwrap().len(),
            1
        );
        assert!(closed(&mut stale));
    }

    #[test]
    fn a_rank_of_another_build_fails_its_launch_by_name() {
        let head = Head::bind(false).unwrap();
        let mut head = lock(&head).unwrap();
        let address = head.address(None).unwrap();

        let generation = head.begin_launch("text");
        let mut foreign = register(&address, "text", 0, generation, "0.0.0-other");
        let error = head
            .collect("text", generation, 1, || Ok(()))
            .err()
            .expect("a rank of another build fails the launch")
            .to_string();
        assert!(
            error.contains("0.0.0-other") && error.contains(VERSION),
            "{error}"
        );
        assert!(closed(&mut foreign));
    }

    #[test]
    fn launchers_register_on_the_tcp_port_and_one_of_another_build_is_refused() {
        let head = Head::bind(true).unwrap();
        let mut head = lock(&head).unwrap();
        let port = head.tcp_address().rsplit_once(':').unwrap().1.to_owned();
        let address = format!("127.0.0.1:{port}");
        let present = |host: &str, version: &str| {
            let mut channel = Channel::connect(&address, Duration::from_secs(5)).unwrap();
            channel
                .send(&Registration::Launcher {
                    host: host.into(),
                    version: version.into(),
                })
                .unwrap();
            channel
        };
        let hosts = ["b".to_owned(), "c".to_owned()];

        let _launcher = present("b", VERSION);
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut presented = Vec::new();
        while presented.is_empty() && Instant::now() < deadline {
            presented = head.take_launchers(&hosts).unwrap();
            head.park().unwrap();
        }
        assert_eq!(presented.len(), 1);
        assert_eq!(presented[0].0, "b");

        let mut foreign = present("c", "0.0.0-other");
        let error = loop {
            match head.take_launchers(&hosts) {
                Ok(_) => {
                    assert!(Instant::now() < deadline, "the launcher was not refused");
                    head.park().unwrap();
                }
                Err(error) => break error.to_string(),
            }
        };
        assert!(
            error.contains("host c") && error.contains("0.0.0-other"),
            "{error}"
        );
        assert!(closed(&mut foreign));
    }

    #[test]
    fn a_rank_that_dies_before_registering_fails_the_launch() {
        let head = Head::bind(false).unwrap();
        let mut head = lock(&head).unwrap();
        let generation = head.begin_launch("text");

        let error = head
            .collect("text", generation, 1, || anyhow::bail!("rank 0 exited"))
            .err()
            .expect("the liveness check fails the launch")
            .to_string();
        assert!(error.contains("rank 0 exited"), "{error}");
    }
}
