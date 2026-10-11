//! The head's registry of per-host launchers.
//!
//! The engine starts no process on another host. Each host of a deployment
//! runs one `uniserve-host` launcher that connects to the head and registers
//! its host identity; the head then sends that launcher the launches of the
//! ranks the placement put on its host, of every worker group, and the
//! launcher spawns them. The registry is built once per deployment and shared
//! by its groups, so a host presents one launcher however many workers place
//! ranks on it; a rank is named to its launcher by worker and rank.
//!
//! A launcher's connection is its liveness signal in both directions. Closing
//! it terminates that host's ranks, which is how an instance stops a host it
//! can no longer reach without adding a heartbeat.
//!
//! The messages are `uniserve_ipc::launcher`'s, which the `uniserve-host`
//! binary shares, carried as frames on the connection the launcher registered
//! on (`registry::RankRegistry`). A `Stop` ends a group's generation before
//! the group is relaunched under the next one. A launcher may report an exit
//! of a stopped rank before it reads the `Stop`, and the head may read that
//! report only after the relaunch, so an exit is attributed to a group only
//! when it names the group's current generation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use uniserve_ipc::Channel;
use uniserve_ipc::launcher::{Instruction, RankExit, RankSpawn, Report};

/// Longest a launcher may take to answer a reservation.
const RESERVATION_TIMEOUT: Duration = Duration::from_secs(10);

/// The registry of one deployment, shared by every group that places a rank
/// on another host.
pub(crate) type Launchers = Arc<Mutex<LauncherRegistry>>;

/// Locks the deployment's registry for one operation.
pub(crate) fn lock(
    registry: &Launchers,
) -> anyhow::Result<std::sync::MutexGuard<'_, LauncherRegistry>> {
    registry
        .lock()
        .map_err(|_| anyhow::anyhow!("the launcher registry is poisoned"))
}

/// One connected launcher.
struct Launcher {
    channel: Channel,
}

/// Rank exits read from the launchers and not yet taken, and the generation
/// each worker group is launched under.
///
/// A launcher reports every group's ranks on one connection, and each group
/// takes its own. Only exits of a group's current generation are kept, so a
/// report of a rank a stop ended is never attributed to its relaunch.
#[derive(Default)]
struct ExitReports {
    /// Each group's current generation; a group absent here has never been
    /// stopped and is in generation zero.
    generations: HashMap<String, u64>,
    /// Exits of each group's current generation, with the host that
    /// reported each, by worker group.
    reported: HashMap<String, Vec<(String, RankExit)>>,
}

impl ExitReports {
    /// Returns the generation `worker_id`'s launches currently carry.
    fn generation(&self, worker_id: &str) -> u64 {
        self.generations.get(worker_id).copied().unwrap_or(0)
    }

    /// Keeps one exit a launcher on `host` reported for its group, or
    /// discards it when it names a generation a stop has ended.
    fn file(&mut self, host: &str, exit: RankExit) {
        if exit.generation != self.generation(&exit.worker_id) {
            tracing::debug!(
                host,
                worker = %exit.worker_id,
                rank = exit.rank,
                generation = exit.generation,
                "discarding the exit of a stopped rank"
            );
            return;
        }
        self.reported
            .entry(exit.worker_id.clone())
            .or_default()
            .push((host.to_owned(), exit));
    }

    /// Takes every kept exit of `worker_id`.
    fn take(&mut self, worker_id: &str) -> Vec<(String, RankExit)> {
        self.reported.remove(worker_id).unwrap_or_default()
    }

    /// Ends `worker_id`'s current generation: its kept exits are discarded,
    /// as is every exit of that generation read later.
    fn end_generation(&mut self, worker_id: &str) {
        *self.generations.entry(worker_id.to_owned()).or_default() += 1;
        self.reported.remove(worker_id);
    }
}

/// The launchers that registered with the head, and the exits they report.
pub(crate) struct LauncherRegistry {
    /// Connected launchers by the host identity each registered.
    hosts: HashMap<String, Launcher>,
    /// Exits read from the launchers, kept for the group each belongs to.
    exits: ExitReports,
}

impl LauncherRegistry {
    /// Builds a registry and waits for every remote host of a deployment to
    /// register its launcher with `head`.
    ///
    /// `host` is this process's own identity; a launcher is awaited for each
    /// other host any of the placements names. A deployment placing every
    /// rank on this host needs no registry.
    pub(crate) fn for_hosts(
        registry: &super::registry::Registry,
        host: &str,
        placements: impl IntoIterator<Item = impl AsRef<[crate::WorkerRank]>>,
        timeout: Duration,
    ) -> anyhow::Result<Option<Launchers>> {
        let mut remote_hosts: Vec<String> = placements
            .into_iter()
            .flat_map(|ranks| {
                ranks
                    .as_ref()
                    .iter()
                    .filter(|rank| rank.node != host)
                    .map(|rank| rank.node.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        remote_hosts.sort();
        remote_hosts.dedup();
        if remote_hosts.is_empty() {
            return Ok(None);
        }

        let mut launchers = Self {
            hosts: HashMap::new(),
            exits: ExitReports::default(),
        };
        tracing::info!(
            address = super::registry::lock(registry)?.tcp_address(),
            hosts = ?remote_hosts,
            "awaiting a launcher for each host this instance does not run on"
        );
        launchers.await_hosts(registry, &remote_hosts, timeout)?;
        Ok(Some(Arc::new(Mutex::new(launchers))))
    }

    /// Returns a host address of this machine that a launcher reached it at.
    ///
    /// The head cannot name itself: its placement identity need not resolve on
    /// another host, and a bound wildcard address names no interface. But a
    /// launcher has already connected to the head, so the connection's local
    /// address is a head address reachable from a launcher host. The
    /// rendezvous address and the head's address handed to remote ranks must
    /// use that address.
    pub(crate) fn reachable_host(&self) -> anyhow::Result<std::net::IpAddr> {
        let launcher = self
            .hosts
            .values()
            .next()
            .context("no launcher has presented a reachable head address")?;
        Ok(launcher
            .channel
            .local_addr()?
            .context("a launcher reached the head over a Unix socket")?
            .ip())
    }

    /// Waits until every named host has registered a launcher.
    ///
    /// A host that never registers fails the launch by name rather than
    /// leaving the instance waiting for ranks that were never started. A
    /// launcher of another build than the head's fails the wait by name.
    pub(crate) fn await_hosts(
        &mut self,
        registry: &super::registry::Registry,
        hosts: &[String],
        timeout: Duration,
    ) -> anyhow::Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            let mut head = super::registry::lock(registry)?;
            for (host, channel) in head.take_launchers()? {
                tracing::info!(host = %host, "a launcher presented its host");
                self.hosts.insert(host, Launcher { channel });
            }
            let Some(missing) = hosts.iter().find(|host| !self.hosts.contains_key(*host)) else {
                return Ok(());
            };
            anyhow::ensure!(
                Instant::now() < deadline,
                "no launcher presented host {missing} before its deadline"
            );
            head.park()?;
        }
    }

    /// Returns the rendezvous for a group whose first rank runs on `host`,
    /// which is another machine.
    ///
    /// The first rank serves the collective store, so the store's address has
    /// to be one of that rank's own host: the address its launcher connected
    /// from, at the port the launcher reserved there. The launcher keeps that
    /// port bound and hands its socket to the first rank it spawns for the
    /// group, so the rendezvous returned here carries no listener.
    pub(crate) fn rendezvous_on(
        &mut self,
        host: &str,
        worker_id: &str,
    ) -> anyhow::Result<super::rendezvous::Rendezvous> {
        let launcher = self
            .hosts
            .get_mut(host)
            .with_context(|| format!("no launcher presented host {host}"))?;
        let address = launcher
            .channel
            .peer_addr()?
            .context("a launcher reached the head over a Unix socket")?
            .ip();
        launcher.channel.send(&Instruction::Reserve {
            worker_id: worker_id.to_owned(),
        })?;

        // The launcher may report rank exits before it answers; those are
        // kept for their groups and the wait continues until the reply.
        let deadline = Instant::now() + RESERVATION_TIMEOUT;
        loop {
            match launcher.channel.try_recv::<Report>() {
                Ok(Some(Report::Reservation(reservation))) => {
                    anyhow::ensure!(
                        reservation.worker_id == worker_id && reservation.port > 0,
                        "invalid rendezvous reservation"
                    );
                    return Ok(super::rendezvous::Rendezvous {
                        address: std::net::SocketAddr::new(address, reservation.port).to_string(),
                        listener: None,
                        listen_fd: Some(reservation.listen_fd),
                    });
                }
                Ok(Some(Report::Exit(exit))) => self.exits.file(host, exit),
                Ok(None) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    anyhow::ensure!(
                        !remaining.is_zero(),
                        "the launcher of host {host} did not answer a rendezvous reservation"
                    );
                    launcher.channel.wait(remaining)?;
                }
                Err(error) => {
                    return Err(error).context("launcher disconnected while reserving rendezvous");
                }
            }
        }
    }

    /// Sends one rank's launch to the launcher that owns its host. `launch`
    /// states the rank's spawn for the generation `worker_id`'s launches
    /// currently carry, which only the registry knows.
    pub(crate) fn spawn_remote(
        &mut self,
        host: &str,
        worker_id: &str,
        launch: impl FnOnce(u64) -> RankSpawn,
    ) -> anyhow::Result<()> {
        let spawn = launch(self.exits.generation(worker_id));
        let launcher = self
            .hosts
            .get_mut(host)
            .with_context(|| format!("no launcher presented host {host}"))?;
        launcher
            .channel
            .send(&Instruction::Spawn(Box::new(spawn)))
            .with_context(|| format!("sending a launch to host {host}"))
    }

    /// Takes every exit of one worker group's running launch its launchers
    /// have reported, keeping the other groups' exits for their own owners.
    ///
    /// The drain is best effort and never blocks: it reads whatever each
    /// connection holds, discards exits of a launch a stop has ended, and
    /// moves on from a launcher at its first receive error or close. A closed
    /// launcher connection is therefore not reported here.
    pub(crate) fn drain_exits(&mut self, worker_id: &str) -> Vec<(String, RankExit)> {
        for (host, launcher) in self.hosts.iter_mut() {
            while let Ok(Some(report)) = launcher.channel.try_recv::<Report>() {
                if let Report::Exit(exit) = report {
                    self.exits.file(host, exit);
                }
            }
        }
        self.exits.take(worker_id)
    }

    /// Stops one worker group's ranks on every launcher, ahead of relaunching
    /// the group; the launchers stay connected for the other groups.
    ///
    /// The group's generation ends first, even if a stop cannot be sent: its
    /// exits already collected are discarded, and so is any exit of the
    /// stopped ranks read later, including one a launcher wrote before it
    /// read the stop. The launcher drops the stopped ranks from its
    /// supervision before terminating them, so a rank still running at the
    /// stop is not reported at all.
    pub(crate) fn stop_worker(&mut self, worker_id: &str) -> anyhow::Result<()> {
        self.exits.end_generation(worker_id);
        let stop = Instruction::Stop {
            worker_id: worker_id.to_owned(),
        };
        for (host, launcher) in self.hosts.iter_mut() {
            launcher
                .channel
                .send(&stop)
                .with_context(|| format!("sending a worker stop to host {host}"))?;
        }
        Ok(())
    }
}

impl Drop for LauncherRegistry {
    /// Stops every host's ranks when the instance stops.
    ///
    /// A launcher terminates its ranks when this connection closes, so the
    /// instruction is a courtesy that lets a reachable host stop cleanly; the
    /// close does the work either way.
    fn drop(&mut self) {
        for (host, launcher) in self.hosts.iter_mut() {
            if launcher.channel.send(&Instruction::Terminate).is_err() {
                tracing::debug!(host = %host, "a launcher was already gone at shutdown");
            }
        }
    }
}

/// One host's launcher, addressed while a rank's launch is being derived.
///
/// The head builds a rank's command the same way whether the rank runs here
/// or elsewhere; this delivers that launch to the host that owns it, taking
/// the environment from the command rather than deriving it a second time.
pub(crate) struct RemoteHost<'a> {
    pub registry: &'a Mutex<LauncherRegistry>,
    pub host: &'a str,
}

impl RemoteHost<'_> {
    /// Sends one rank's launch to this host's launcher. The rank connects to
    /// the head at `head`.
    pub(crate) fn deliver(
        &mut self,
        rank: u32,
        python: &std::path::Path,
        worker_id: &str,
        head: &str,
        command: &std::process::Command,
    ) -> anyhow::Result<()> {
        let environment = command
            .get_envs()
            .filter_map(|(name, value)| {
                Some((
                    name.to_str()?.to_owned(),
                    value
                        .and_then(|value| value.to_str())
                        .unwrap_or("")
                        .to_owned(),
                ))
            })
            .collect();
        let python = python
            .to_str()
            .context("the interpreter path is not valid text")?;
        let mut registry = self
            .registry
            .lock()
            .map_err(|_| anyhow::anyhow!("the launcher registry is poisoned"))?;
        registry.spawn_remote(self.host, worker_id, |generation| RankSpawn {
            rank,
            worker_id: worker_id.to_owned(),
            generation,
            python: python.to_owned(),
            head: head.to_owned(),
            environment,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{LauncherRegistry, RankSpawn};
    use crate::worker::registry::{RankRegistry, Registry, lock};
    use std::collections::HashMap;
    use std::time::{Duration, Instant};
    use uniserve_ipc::launcher::{Instruction, RankExit, Report};
    use uniserve_ipc::{Channel, Registration, VERSION};

    /// Binds a head and connects a launcher of host `b` to its TCP port.
    fn head_and_launcher() -> (Registry, Channel) {
        let head = RankRegistry::bind(true).expect("the head binds");
        let port = lock(&head)
            .expect("the head locks")
            .tcp_address()
            .rsplit_once(':')
            .expect("a port")
            .1
            .to_owned();
        let mut launcher =
            Channel::connect(&format!("127.0.0.1:{port}")).expect("the launcher connects");
        launcher
            .send(&Registration::Launcher {
                host: "b".into(),
                version: VERSION.into(),
            })
            .expect("the launcher registers");
        (head, launcher)
    }

    /// A registry in which host `b` has presented.
    fn registry(head: &Registry) -> LauncherRegistry {
        LauncherRegistry::for_hosts(
            head,
            "a",
            [[crate::WorkerRank {
                node: "b".into(),
                device: "cpu".into(),
            }]],
            Duration::from_secs(5),
        )
        .expect("host b presents")
        .and_then(|registry| std::sync::Arc::try_unwrap(registry).ok())
        .and_then(|registry| registry.into_inner().ok())
        .expect("a registry for host b")
    }

    /// Waits for the launcher's next message from the head.
    fn receive<M: serde::de::DeserializeOwned>(launcher: &mut Channel) -> M {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(message) = launcher.try_recv().expect("the head is connected") {
                return message;
            }
            assert!(Instant::now() < deadline, "no message arrived");
            launcher
                .wait(Duration::from_millis(20))
                .expect("the launcher waits");
        }
    }

    /// Launches one rank of `worker_id` on host `b`.
    fn spawn(registry: &mut LauncherRegistry, worker_id: &str, rank: u32) {
        registry
            .spawn_remote("b", worker_id, |generation| RankSpawn {
                rank,
                worker_id: worker_id.to_owned(),
                generation,
                python: "python3".to_owned(),
                head: "127.0.0.1:1".to_owned(),
                environment: HashMap::new(),
            })
            .expect("the rank is launched");
    }

    /// An exit of the rank `spawn` launched.
    fn exit(spawn: &RankSpawn, status: &str) -> Report {
        Report::Exit(RankExit {
            worker_id: spawn.worker_id.clone(),
            generation: spawn.generation,
            rank: spawn.rank,
            status: status.to_owned(),
        })
    }

    /// Returns the statuses of the first nonempty drain of `worker_id`'s
    /// exits within five seconds, or none.
    fn await_exits(registry: &mut LauncherRegistry, worker_id: &str) -> Vec<String> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let exits = registry.drain_exits(worker_id);
            if !exits.is_empty() || Instant::now() >= deadline {
                return exits.into_iter().map(|(_, exit)| exit.status).collect();
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// One launcher serves every worker group placed on its host: the exits
    /// it reports are routed to the group they belong to, and a stop names
    /// the group whose ranks it ends.
    #[test]
    fn exits_are_routed_to_their_worker_and_a_stop_names_its_worker() {
        let (head, mut launcher) = head_and_launcher();
        let mut registry = registry(&head);

        // Both ranks are launched, and then both exit.
        for (worker, rank) in [("model", 5), ("host", 1)] {
            spawn(&mut registry, worker, rank);
        }
        for status in ["exit status: 1", "exit status: 2"] {
            let Instruction::Spawn(launched) = receive(&mut launcher) else {
                panic!("the head launches a rank");
            };
            launcher
                .send(&exit(&launched, status))
                .expect("the launcher reports an exit");
        }

        // Both exits arrive on one connection; each group takes its own and
        // the other's is kept for its owner.
        assert_eq!(await_exits(&mut registry, "model"), ["exit status: 1"]);
        let host_exits = registry.drain_exits("host");
        assert_eq!(host_exits.len(), 1);
        assert_eq!(
            (
                host_exits[0].0.as_str(),
                host_exits[0].1.worker_id.as_str(),
                host_exits[0].1.rank
            ),
            ("b", "host", 1)
        );
        assert!(registry.drain_exits("model").is_empty());

        registry.stop_worker("host").expect("the stop is sent");
        assert_eq!(
            receive::<Instruction>(&mut launcher),
            Instruction::Stop {
                worker_id: "host".into()
            }
        );
    }

    /// A stop ends a group's generation: an exit its launcher reported for a
    /// stopped rank before reading the stop, and which the head has not read
    /// when it relaunches the group, is not attributed to the relaunch,
    /// while an exit of a relaunched rank is.
    #[test]
    fn an_exit_reported_before_a_stop_is_not_attributed_to_the_relaunch() {
        let (head, mut launcher) = head_and_launcher();
        let mut registry = registry(&head);

        // The failed rank's exit is written before the stop is read.
        spawn(&mut registry, "model", 5);
        let Instruction::Spawn(failed) = receive(&mut launcher) else {
            panic!("the head launches a rank");
        };
        launcher
            .send(&exit(&failed, "signal: 9 (SIGKILL)"))
            .expect("the launcher reports the failed rank");

        registry.stop_worker("model").expect("the stop is sent");
        spawn(&mut registry, "model", 5);
        assert_eq!(
            receive::<Instruction>(&mut launcher),
            Instruction::Stop {
                worker_id: "model".into()
            }
        );
        let Instruction::Spawn(relaunched) = receive(&mut launcher) else {
            panic!("the head relaunches the rank");
        };
        launcher
            .send(&exit(&relaunched, "exit status: 1"))
            .expect("the launcher reports the relaunched rank");

        assert_eq!(await_exits(&mut registry, "model"), ["exit status: 1"]);
    }
}
