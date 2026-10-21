//! The engine's listener and the ranks that registered there.
//!
//! The engine listens on one Unix socket path and one TCP port for the whole
//! deployment. A rank connects to the engine when it starts, through the Unix
//! socket on the engine's node and through TCP from any other node, and its
//! first frame is its `Registration`: worker group, rank, launch generation
//! and build version. [`WorkerRegistry::begin_launch`] numbers each launch of
//! a group, and [`collect`] gathers the registrations of the
//! group's current launch and hands each rank's connection to that launch,
//! which answers with the rank's launch descriptor. The connection then
//! serves the rank for its lifetime, and its close is the rank's death.
//!
//! A registration names the launch it belongs to because a group is
//! relaunched after it loses a rank, and a rank of the launch that ended
//! must not join the relaunch. Such a rank is refused by closing its
//! connection, which ends it; the relaunch goes on.
//!
//! A rank therefore accepts no inbound connection but its group's peer mesh
//! and its collective store.

use std::collections::{BTreeMap, HashMap};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use ipc::{Channel, Listener, Registration, VERSION};

/// Deadline for every rank of one launch to register, measured from the
/// start of `collect`, which runs after every rank of the group
/// has been launched.
///
/// A rank registers before it builds its model, so this covers process start
/// and imports rather than weight loading.
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(300);

/// Interval between checks that the ranks of a launch are still alive while
/// some of them have not registered.
const ACCEPT_INTERVAL: Duration = Duration::from_millis(20);

/// The deployment's registry, shared by its worker groups.
pub(crate) type Registry = Arc<Mutex<WorkerRegistry>>;

/// Locks the registry, reporting a poisoned lock as an error.
pub(crate) fn lock(
    registry: &Mutex<WorkerRegistry>,
) -> anyhow::Result<std::sync::MutexGuard<'_, WorkerRegistry>> {
    registry
        .lock()
        .map_err(|_| anyhow::anyhow!("the rank registry is poisoned"))
}

/// A rank of a group's current launch that has registered and that the
/// launch has not taken yet.
struct RegisteredWorker {
    worker_id: String,
    rank: u32,
    channel: Channel,
}

/// The address ranks connect to, and those that have registered there.
pub(crate) struct WorkerRegistry {
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
    ranks: Vec<RegisteredWorker>,
}

impl WorkerRegistry {
    /// Binds the Unix socket path and the TCP port ranks connect to.
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

    /// Moves the registrations of `worker_id`'s current launch that have
    /// arrived into `ranks`, by rank, and leaves the other groups'.
    ///
    /// Fails when a registration states another build than the head's, and
    /// when one names a rank outside `0..expected` or one already in `ranks`.
    fn take_registered(
        &mut self,
        worker_id: &str,
        expected: usize,
        ranks: &mut BTreeMap<usize, Channel>,
    ) -> anyhow::Result<()> {
        self.serve()?;
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
        Ok(())
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
            let Registration::Worker {
                worker_id,
                rank,
                generation,
                version,
            } = registration;
            anyhow::ensure!(
                version == VERSION,
                "rank {rank} of worker {worker_id} is build {version} and the engine is build \
                 {VERSION}; reinstall so that both run the same build"
            );
            let current = self.generations.get(&worker_id).copied();
            if current != Some(generation) {
                // Dropping the connection is the refusal: the rank exits when
                // its connection closes.
                tracing::warn!(
                    worker = %worker_id,
                    rank,
                    generation,
                    current = ?current,
                    "refusing a rank of a launch that is not its group's current one"
                );
                continue;
            }
            self.ranks.push(RegisteredWorker {
                worker_id,
                rank,
                channel,
            });
        }
        Ok(())
    }
}

/// Collects the connection of every rank of `worker_id`'s current launch,
/// ordered by rank.
///
/// The registry is locked only to read what has arrived, never across the
/// wait: other groups begin launches, start standbys and collect their own
/// ranks meanwhile, from their own threads. The wait parks on the listener;
/// a registration on a connection the registry has already accepted is read
/// at the next interval.
///
/// `alive` is consulted whenever the launch is still incomplete, so a rank
/// that dies before registering fails the launch by name instead of
/// consuming the deadline.
///
/// Fails on an error from `alive`, when the deadline passes with ranks
/// unregistered (named in the error), when a registration states another
/// build than the head's, and when one names a rank outside `0..expected`
/// or one that already registered. Connections of other worker groups are
/// kept for their own launches, and a rank of an earlier launch of this
/// group is refused without failing this one.
pub(crate) fn collect(
    registry: &Mutex<WorkerRegistry>,
    worker_id: &str,
    expected: usize,
    mut alive: impl FnMut() -> anyhow::Result<()>,
) -> anyhow::Result<Vec<Channel>> {
    let deadline = Instant::now() + REGISTRATION_TIMEOUT;
    let mut ranks = BTreeMap::<usize, Channel>::new();
    loop {
        // The listener lives as long as the registry, which the caller
        // holds, so its descriptors stay valid once the lock is released.
        let descriptors = {
            let mut head = lock(registry)?;
            head.take_registered(worker_id, expected, &mut ranks)?;
            head.listener.poll_fds()
        };
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
        super::park_descriptors(&descriptors, ACCEPT_INTERVAL)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Connects to the head and registers as one rank of launch `generation`.
    fn register(address: &str, worker: &str, rank: u32, generation: u64, version: &str) -> Channel {
        let mut channel = Channel::connect(address).unwrap();
        channel
            .send(&Registration::Worker {
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
    fn a_launch_collects_its_own_workers_over_both_addresses_in_rank_order() {
        let shared = WorkerRegistry::bind(false).unwrap();
        let registry = || lock(&shared).unwrap();
        let unix = registry().address(None).unwrap();
        let tcp = registry()
            .address(Some(Ipv4Addr::LOCALHOST.into()))
            .unwrap();
        assert!(unix.starts_with("unix:") && !tcp.starts_with("unix:"));

        let image = registry().begin_launch("image");
        let text = registry().begin_launch("text");
        let _other = register(&unix, "image", 0, image, VERSION);
        let mut second = register(&tcp, "text", 1, text, VERSION);
        let mut first = register(&unix, "text", 0, text, VERSION);

        // Each collected connection answers the rank that registered on it.
        let mut ranks = collect(&shared, "text", 2, || Ok(())).unwrap();
        assert_eq!(ranks.len(), 2);
        for (rank, channel) in ranks.iter_mut().enumerate() {
            channel
                .send(&ipc::WorkerRequest::Info {
                    message_id: Some(rank as u64),
                })
                .unwrap();
        }
        for (rank, peer) in [&mut first, &mut second].into_iter().enumerate() {
            let deadline = Instant::now() + Duration::from_secs(5);
            let request = loop {
                if let Some(request) = peer.try_recv::<ipc::WorkerRequest>().unwrap() {
                    break request;
                }
                assert!(Instant::now() < deadline);
                peer.wait(Duration::from_millis(20)).unwrap();
            };
            assert_eq!(request.message_id(), Some(rank as u64));
        }

        // The other group's rank stayed for its own launch.
        assert_eq!(collect(&shared, "image", 1, || Ok(())).unwrap().len(), 1);
    }

    /// A relaunch takes only its own ranks. A rank of the launch it replaced
    /// is refused whether it registered before the relaunch began or after,
    /// and neither counts as a rank of the relaunch.
    #[test]
    fn a_worker_of_an_earlier_launch_is_refused_and_the_relaunch_completes() {
        let shared = WorkerRegistry::bind(false).unwrap();
        let registry = || lock(&shared).unwrap();
        let address = registry().address(None).unwrap();

        // A rank of the first launch registers and is read while that launch
        // is current, here by another group's launch, which reads every
        // registration that has arrived.
        let first = registry().begin_launch("text");
        let image = registry().begin_launch("image");
        let mut early = register(&address, "text", 0, first, VERSION);
        let _image = register(&address, "image", 0, image, VERSION);
        assert_eq!(collect(&shared, "image", 1, || Ok(())).unwrap().len(), 1);

        let relaunch = registry().begin_launch("text");
        let mut late = register(&address, "text", 1, first, VERSION);
        let _current = [
            register(&address, "text", 0, relaunch, VERSION),
            register(&address, "text", 1, relaunch, VERSION),
        ];
        assert_eq!(collect(&shared, "text", 2, || Ok(())).unwrap().len(), 2);
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
    fn a_worker_of_another_build_fails_the_wait_by_name() {
        let shared = WorkerRegistry::bind(true).unwrap();
        let registry = || lock(&shared).unwrap();

        // Another build of the same package version.
        let package = VERSION.split_once('+').unwrap().0;
        let other = format!("{package}+0123456789abcdef");
        assert_ne!(other, VERSION);

        let generation = registry().begin_launch("text");
        let _foreign = register(
            &registry().address(None).unwrap(),
            "text",
            0,
            generation,
            &other,
        );
        let error = collect(&shared, "text", 1, || Ok(()))
            .err()
            .expect("a rank of another build fails the launch")
            .to_string();
        assert!(
            error.contains("rank 0 of worker text")
                && error.contains(&other)
                && error.contains(VERSION),
            "{error}"
        );
    }

    #[test]
    fn a_worker_outside_the_group_and_a_worker_registered_twice_fail_the_launch() {
        let shared = WorkerRegistry::bind(false).unwrap();
        let registry = || lock(&shared).unwrap();
        let address = registry().address(None).unwrap();

        let text = registry().begin_launch("text");
        let _outside = register(&address, "text", 2, text, VERSION);
        let error = collect(&shared, "text", 2, || Ok(()))
            .err()
            .expect("a rank outside the group fails the launch")
            .to_string();
        assert!(error.contains("rank 2"), "{error}");

        let image = registry().begin_launch("image");
        let _first = register(&address, "image", 0, image, VERSION);
        let _again = register(&address, "image", 0, image, VERSION);
        let error = collect(&shared, "image", 2, || Ok(()))
            .err()
            .expect("a rank registered twice fails the launch")
            .to_string();
        assert!(error.contains("registered twice"), "{error}");
    }

    /// A connection that never registers, and one that sends something that
    /// is not a registration, do not keep a launch from completing.
    #[test]
    fn a_launch_completes_beside_connections_that_never_register() {
        let shared = WorkerRegistry::bind(false).unwrap();
        let registry = || lock(&shared).unwrap();
        let address = registry().address(None).unwrap();

        let _silent = Channel::connect(&address).unwrap();
        let mut stray = Channel::connect(&address).unwrap();
        stray.send(&ipc::WorkerRequest::info()).unwrap();

        let generation = registry().begin_launch("text");
        let _rank = register(&address, "text", 0, generation, VERSION);
        assert_eq!(collect(&shared, "text", 1, || Ok(())).unwrap().len(), 1);
        assert!(closed(&mut stray), "the stray connection stayed open");
    }

    #[test]
    fn a_worker_that_dies_before_registering_fails_the_launch() {
        let shared = WorkerRegistry::bind(false).unwrap();
        let registry = || lock(&shared).unwrap();

        registry().begin_launch("text");
        let error = collect(&shared, "text", 1, || anyhow::bail!("rank 0 exited"))
            .err()
            .expect("the liveness check fails the launch")
            .to_string();
        assert!(error.contains("rank 0 exited"), "{error}");
    }
}
