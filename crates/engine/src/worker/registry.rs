//! The head's listener and the ranks and launchers that registered there.
//!
//! The head listens on one Unix socket path and one TCP port for the whole
//! deployment. A rank connects to the head when it starts, through the Unix
//! socket on the head's host and through TCP from any other host, and its
//! first frame is its `Registration`: worker group, rank, launch generation
//! and build version. [`RankRegistry::begin_launch`] numbers each launch of
//! a group, and [`RankRegistry::collect`] gathers the registrations of the
//! group's current launch and hands each rank's connection to that launch,
//! which answers with the rank's launch descriptor. The connection then
//! serves the rank for its lifetime, and its close is the rank's death.
//!
//! A registration names the launch it belongs to because a group is
//! relaunched after it loses a rank, and a rank of the launch that ended
//! must not join the relaunch. Such a rank is refused by closing its
//! connection, which ends it; the relaunch goes on.
//!
//! The launcher of another host connects to the same TCP port and registers
//! its host identity; [`RankRegistry::take_launchers`] hands its connection
//! to the launcher registry (`launcher`).
//!
//! A host that runs ranks therefore accepts no inbound connection.

use std::collections::{BTreeMap, HashMap};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use uniserve_ipc::{Channel, Listener, Registration, VERSION};

/// Deadline for every rank of one launch to register, measured from the
/// start of `RankRegistry::collect`, which runs after every rank of the group
/// has been launched.
///
/// A rank registers before it builds its model, so this covers process start
/// and imports rather than weight loading.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(300);

/// Interval between checks that the ranks of a launch are still alive while
/// some of them have not registered.
const ACCEPT_INTERVAL: Duration = Duration::from_millis(20);

/// The deployment's registry, shared by its worker groups.
pub(crate) type Registry = Arc<Mutex<RankRegistry>>;

/// Locks the registry, reporting a poisoned lock as an error.
pub(crate) fn lock(
    registry: &Mutex<RankRegistry>,
) -> anyhow::Result<std::sync::MutexGuard<'_, RankRegistry>> {
    registry
        .lock()
        .map_err(|_| anyhow::anyhow!("the rank registry is poisoned"))
}

/// A rank of a group's current launch that has registered and that the
/// launch has not taken yet.
struct RegisteredRank {
    worker_id: String,
    rank: u32,
    channel: Channel,
}

/// The address ranks and launchers connect to, and those that have
/// registered there.
pub(crate) struct RankRegistry {
    listener: Listener,
    /// Holds the Unix socket's path; dropping it removes the directory.
    _directory: tempfile::TempDir,
    /// The generation of each worker group's current launch, which every
    /// rank of that launch states when it registers.
    generations: HashMap<String, u64>,
    /// Accepted connections whose registration has not arrived.
    connecting: Vec<Channel>,
    /// Registered ranks of the groups' current launches that the launches
    /// have not taken yet.
    ranks: Vec<RegisteredRank>,
    /// Registered launchers the launcher registry has not taken yet, by host.
    launchers: Vec<(String, Channel)>,
}

impl RankRegistry {
    /// Binds the Unix socket path and the TCP port ranks and launchers
    /// connect to.
    ///
    /// `routable` says whether some rank runs on another host: the TCP port
    /// then binds every interface, and loopback otherwise, since a rank on
    /// another host cannot reach a loopback address and a deployment on one
    /// host does not need a routable one.
    pub(crate) fn bind(routable: bool) -> anyhow::Result<Registry> {
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
        let listener = Listener::bind(&[&unix, &tcp]).context("binding the head's addresses")?;
        Ok(Arc::new(Mutex::new(Self {
            listener,
            _directory: directory,
            generations: HashMap::new(),
            connecting: Vec::new(),
            ranks: Vec::new(),
            launchers: Vec::new(),
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

    /// Returns the TCP address the head listens at, as it was bound.
    pub(crate) fn tcp_address(&self) -> &str {
        &self.listener.addresses()[1]
    }

    /// Starts a launch of `worker_id` and returns its generation, which the
    /// launch's ranks are started with and state when they register.
    ///
    /// A group's first launch is generation 1 and each relaunch the next. The
    /// group's earlier launch ends here: a registration of it that no launch
    /// took is dropped, which closes its connection and so ends that rank,
    /// and one that arrives later is refused the same way.
    pub(crate) fn begin_launch(&mut self, worker_id: &str) -> u64 {
        self.ranks.retain(|rank| rank.worker_id != worker_id);

        let generation = self.generations.entry(worker_id.to_owned()).or_default();
        *generation += 1;
        *generation
    }

    /// Collects the connection of every rank of `worker_id`'s current launch,
    /// ordered by rank.
    ///
    /// `alive` is consulted whenever the launch is still incomplete, so a
    /// rank that dies before registering fails the launch by name instead of
    /// consuming the deadline.
    ///
    /// Fails on an error from `alive`, when the deadline passes with ranks
    /// unregistered (named in the error), when a registration states another
    /// build than the head's, and when one names a rank outside
    /// `0..expected` or one that already registered. Connections of other
    /// worker groups are kept for their own launches, and a rank of an
    /// earlier launch of this group is refused without failing this one.
    pub(crate) fn collect(
        &mut self,
        worker_id: &str,
        expected: usize,
        mut alive: impl FnMut() -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<Channel>> {
        let deadline = Instant::now() + REGISTRATION_TIMEOUT;
        let mut ranks = BTreeMap::<usize, Channel>::new();
        loop {
            self.serve()?;

            // Take this group's registrations and leave the other groups'.
            let mut index = 0;
            while index < self.ranks.len() {
                if self.ranks[index].worker_id != worker_id {
                    index += 1;
                    continue;
                }
                let registered = self.ranks.swap_remove(index);
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

    /// Takes the connection of every launcher that has registered since the
    /// last call, with the host identity each registered.
    ///
    /// Fails when a registration states another build than the head's.
    pub(crate) fn take_launchers(&mut self) -> anyhow::Result<Vec<(String, Channel)>> {
        self.serve()?;
        Ok(std::mem::take(&mut self.launchers))
    }

    /// Waits a short interval for a new connection or a registration.
    pub(crate) fn park(&self) -> anyhow::Result<()> {
        let mut descriptors = self.listener.poll_fds();
        for channel in &self.connecting {
            descriptors.extend(channel.poll_fds());
        }
        super::park_descriptors(&descriptors, ACCEPT_INTERVAL)
    }

    /// Accepts what is waiting and reads the registrations that have arrived.
    ///
    /// A connection that fails or sends something else is dropped, which
    /// closes it, and so is a rank's registration that names another launch
    /// than its group's current one. A registration of another build than
    /// this one fails the call by naming both: the head and whatever
    /// registers with it are installed separately, and a deployment that
    /// mixes builds cannot run.
    fn serve(&mut self) -> anyhow::Result<()> {
        while let Some(channel) = self
            .listener
            .try_accept()
            .context("accepting a connection to the head")?
        {
            self.connecting.push(channel);
        }

        let mut index = 0;
        while index < self.connecting.len() {
            let registration = match self.connecting[index].try_recv::<Registration>() {
                Ok(None) => {
                    index += 1;
                    continue;
                }
                Ok(Some(registration)) => registration,
                Err(error) => {
                    tracing::warn!(%error, "a connection to the head sent no registration");
                    self.connecting.swap_remove(index);
                    continue;
                }
            };
            let channel = self.connecting.swap_remove(index);
            if registration.version() != VERSION {
                let who = match &registration {
                    Registration::Rank {
                        worker_id, rank, ..
                    } => format!("rank {rank} of worker {worker_id}"),
                    Registration::Launcher { host, .. } => format!("the launcher of host {host}"),
                };
                anyhow::bail!(
                    "{who} is build {} and the head is build {VERSION}; reinstall so that both \
                     run the same build",
                    registration.version()
                );
            }
            match registration {
                Registration::Rank {
                    worker_id,
                    rank,
                    generation,
                    ..
                } => {
                    let current = self.generations.get(&worker_id).copied();
                    if current != Some(generation) {
                        // Dropping the connection is the refusal: the rank
                        // exits when its connection closes.
                        tracing::warn!(
                            worker = %worker_id,
                            rank,
                            generation,
                            current = ?current,
                            "refusing a rank of a launch that is not its group's current one"
                        );
                        continue;
                    }
                    self.ranks.push(RegisteredRank {
                        worker_id,
                        rank,
                        channel,
                    });
                }
                Registration::Launcher { host, .. } => self.launchers.push((host, channel)),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Connects to the head and registers as one rank of launch `generation`.
    fn register(address: &str, worker: &str, rank: u32, generation: u64, version: &str) -> Channel {
        let mut channel = Channel::connect(address).unwrap();
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
            if channel.wait(Duration::from_millis(20)).is_err() {
                return true;
            }
        }
        false
    }

    #[test]
    fn a_launch_collects_its_own_ranks_over_both_addresses_in_rank_order() {
        let registry = RankRegistry::bind(false).unwrap();
        let mut registry = lock(&registry).unwrap();
        let unix = registry.address(None).unwrap();
        let tcp = registry.address(Some(Ipv4Addr::LOCALHOST.into())).unwrap();
        assert!(unix.starts_with("unix:") && !tcp.starts_with("unix:"));

        let image = registry.begin_launch("image");
        let text = registry.begin_launch("text");
        let _other = register(&unix, "image", 0, image, VERSION);
        let mut second = register(&tcp, "text", 1, text, VERSION);
        let mut first = register(&unix, "text", 0, text, VERSION);

        // Each collected connection answers the rank that registered on it.
        let mut ranks = registry.collect("text", 2, || Ok(())).unwrap();
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
        assert_eq!(registry.collect("image", 1, || Ok(())).unwrap().len(), 1);
    }

    /// A relaunch takes only its own ranks. A rank of the launch it replaced
    /// is refused whether it registered before the relaunch began or after,
    /// and neither counts as a rank of the relaunch.
    #[test]
    fn a_rank_of_an_earlier_launch_is_refused_and_the_relaunch_completes() {
        let registry = RankRegistry::bind(false).unwrap();
        let mut registry = lock(&registry).unwrap();
        let address = registry.address(None).unwrap();

        // A rank of the first launch registers and is read while that launch
        // is current, here by another group's launch, which reads every
        // registration that has arrived.
        let first = registry.begin_launch("text");
        let image = registry.begin_launch("image");
        let mut early = register(&address, "text", 0, first, VERSION);
        let _image = register(&address, "image", 0, image, VERSION);
        assert_eq!(registry.collect("image", 1, || Ok(())).unwrap().len(), 1);

        let relaunch = registry.begin_launch("text");
        let mut late = register(&address, "text", 1, first, VERSION);
        let _current = [
            register(&address, "text", 0, relaunch, VERSION),
            register(&address, "text", 1, relaunch, VERSION),
        ];
        assert_eq!(registry.collect("text", 2, || Ok(())).unwrap().len(), 2);
        assert!(
            closed(&mut early),
            "a rank of the ended launch kept its connection"
        );
        assert!(
            closed(&mut late),
            "a rank of the ended launch was not refused"
        );
    }

    #[test]
    fn a_rank_or_a_launcher_of_another_build_fails_the_wait_by_name() {
        let registry = RankRegistry::bind(true).unwrap();
        let mut registry = lock(&registry).unwrap();

        // Another build of the same package version.
        let package = VERSION.split_once('+').unwrap().0;
        let other = format!("{package}+0123456789abcdef");
        assert_ne!(other, VERSION);

        let generation = registry.begin_launch("text");
        let _foreign = register(
            &registry.address(None).unwrap(),
            "text",
            0,
            generation,
            &other,
        );
        let error = registry
            .collect("text", 1, || Ok(()))
            .err()
            .expect("a rank of another build fails the launch")
            .to_string();
        assert!(
            error.contains("rank 0 of worker text")
                && error.contains(&other)
                && error.contains(VERSION),
            "{error}"
        );

        let port = registry
            .tcp_address()
            .rsplit_once(':')
            .unwrap()
            .1
            .to_owned();
        let mut launcher = Channel::connect(&format!("127.0.0.1:{port}")).unwrap();
        launcher
            .send(&Registration::Launcher {
                host: "b".into(),
                version: other.clone(),
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let error = loop {
            match registry.take_launchers() {
                Ok(_) => {
                    assert!(Instant::now() < deadline, "the launcher was not refused");
                    registry.park().unwrap();
                }
                Err(error) => break error.to_string(),
            }
        };
        assert!(
            error.contains("host b") && error.contains(&other),
            "{error}"
        );
    }

    #[test]
    fn a_launcher_registers_on_the_tcp_port() {
        let registry = RankRegistry::bind(true).unwrap();
        let mut registry = lock(&registry).unwrap();
        let port = registry
            .tcp_address()
            .rsplit_once(':')
            .unwrap()
            .1
            .to_owned();
        let mut launcher = Channel::connect(&format!("127.0.0.1:{port}")).unwrap();
        launcher
            .send(&Registration::Launcher {
                host: "b".into(),
                version: VERSION.into(),
            })
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        let mut presented = Vec::new();
        while presented.is_empty() && Instant::now() < deadline {
            presented = registry.take_launchers().unwrap();
            registry.park().unwrap();
        }
        assert_eq!(presented.len(), 1);
        assert_eq!(presented[0].0, "b");
    }

    #[test]
    fn a_rank_outside_the_group_and_a_rank_registered_twice_fail_the_launch() {
        let registry = RankRegistry::bind(false).unwrap();
        let mut registry = lock(&registry).unwrap();
        let address = registry.address(None).unwrap();

        let text = registry.begin_launch("text");
        let _outside = register(&address, "text", 2, text, VERSION);
        let error = registry
            .collect("text", 2, || Ok(()))
            .err()
            .expect("a rank outside the group fails the launch")
            .to_string();
        assert!(error.contains("rank 2"), "{error}");

        let image = registry.begin_launch("image");
        let _first = register(&address, "image", 0, image, VERSION);
        let _again = register(&address, "image", 0, image, VERSION);
        let error = registry
            .collect("image", 2, || Ok(()))
            .err()
            .expect("a rank registered twice fails the launch")
            .to_string();
        assert!(error.contains("registered twice"), "{error}");
    }

    /// A connection that never registers, and one that sends something that
    /// is not a registration, do not keep a launch from completing.
    #[test]
    fn a_launch_completes_beside_connections_that_never_register() {
        let registry = RankRegistry::bind(false).unwrap();
        let mut registry = lock(&registry).unwrap();
        let address = registry.address(None).unwrap();

        let _silent = Channel::connect(&address).unwrap();
        let mut stray = Channel::connect(&address).unwrap();
        stray.send(&uniserve_ipc::WorkerRequest::info()).unwrap();

        let generation = registry.begin_launch("text");
        let _rank = register(&address, "text", 0, generation, VERSION);
        assert_eq!(registry.collect("text", 1, || Ok(())).unwrap().len(), 1);
        assert!(closed(&mut stray), "the stray connection stayed open");
    }

    #[test]
    fn a_rank_that_dies_before_registering_fails_the_launch() {
        let registry = RankRegistry::bind(false).unwrap();
        let mut registry = lock(&registry).unwrap();

        registry.begin_launch("text");
        let error = registry
            .collect("text", 1, || anyhow::bail!("rank 0 exited"))
            .err()
            .expect("the liveness check fails the launch")
            .to_string();
        assert!(error.contains("rank 0 exited"), "{error}");
    }
}
