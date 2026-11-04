//! The epoch layer's [`Roster`] over the daemon's supervision
//! (`specs/worker.md` §7.4, §7.3, §2.5).
//!
//! One adapter owns the deployment's main processes for the daemon's life: it
//! starts and stops them through the leader service, commands the process of
//! one launch generation, serves the frontend, and turns what the processes
//! report into the switch's events. It is also the writer of the node's
//! process table (`GET /admin/config`, §2.5): every launch is one row, from
//! `Starting` at its spawn to `Serving`, `Parked`, `Stopped` or `Failed`.
//!
//! A launch generation is the identity of one child; the model generation
//! travels inside each [`process::Command`]. The switch's events carry the
//! launch generation, so the adapter binds a connection to its launch by the
//! generation the connection's first `Parked`/`Promoted` states: reports
//! arrive from the peer that sent them, never from a launch.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use proto::process::{self, Command, Exit, Report, Request, Spawn};

use crate::epoch::{Epochs, ManagedProcess, ProcessState, Role};
use crate::supervision::ProcessCommands;
use crate::updates::switch::{Event, Roster};

/// How long a request to the deployment's daemons waits for its answer: a
/// spawn starts a process, and a claim starts a Storage Server when the node
/// has none.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// A peer that greeted the leader: what it runs, and whether it is the
/// frontend rather than a main process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Greeting {
    /// The build the peer reported in its `Hello`.
    pub(crate) build: String,
    /// The frontend process, which the epoch layer serves an epoch.
    pub(crate) frontend: bool,
}

/// The epoch layer's ends of the daemon's supervision: requests down, and
/// what the daemon sees up.
pub(crate) struct EpochLink {
    /// Requests to the deployment's daemons, each with where its answer goes.
    pub(crate) requests: Sender<(Request, Sender<process::Reply>)>,
    /// Every peer that greeted the leader.
    pub(crate) greetings: Receiver<(u64, Greeting)>,
    /// Every supervised process that exited.
    pub(crate) exits: Receiver<Exit>,
    /// Commands to a connected peer and the reports it sends.
    pub(crate) commands: ProcessCommands,
}

/// One launch the epoch layer started.
struct Launch {
    /// The node it runs on.
    host: String,
    /// The worker group and rank it is one process of.
    group: String,
    rank: u64,
    /// The epoch it serves, once its drain named it.
    epoch: Option<u64>,
    /// Its process id, once the node started it.
    pid: Option<u32>,
}

/// What the epoch layer commands the deployment's daemons with, and what it
/// hears back, as [`crate::updates::switch::Roster`].
pub(crate) struct ProcessRoster {
    epochs: Arc<Epochs>,
    /// The deployment's name, which every claim states.
    deployment: String,
    /// The Storage Server arguments of the deployment's KV storage connector.
    server_arguments: Vec<String>,
    link: EpochLink,
    /// The launches by launch generation. The generation is unique per worker
    /// group and rank, and the epoch layer runs one main-process group, so a
    /// number names one launch.
    launches: BTreeMap<u64, Launch>,
    /// Which launch a connected peer runs, once its report named it.
    engines: BTreeMap<u64, u64>,
    /// The build each greeted peer reported, until the launch it runs is
    /// known.
    builds: BTreeMap<u64, String>,
    /// The peers whose `Hello` the switch was told about.
    greeted: BTreeSet<u64>,
    /// The frontend's connection, once it greeted.
    frontend: Option<u64>,
    /// The request socket the frontend submits to, as the switch stated it.
    requests: String,
    /// Launches the epoch layer stopped: their exit is not a failure.
    stopping: BTreeSet<u64>,
    /// A resume's fresh generation, waiting for the child to park so it can be
    /// promoted: the launch generation the spawn allocated and the epoch it
    /// serves.
    resuming: BTreeMap<u64, u64>,
    /// The epoch a fresh generation is being started for: `spawn` allocates
    /// the generation itself, so the epoch waits here until that number is
    /// known.
    resuming_epoch: Option<u64>,
    /// How the loop last started a main process: the executable and its
    /// arguments, for the fresh generation a restore may need.
    start: Option<(
        String,
        crate::engine::launch::Launch,
        std::collections::HashMap<String, String>,
    )>,
    /// Events the adapter raised, ahead of the ones still on the wire.
    queued: VecDeque<Event>,
    next_request: u64,
}

impl ProcessRoster {
    /// The adapter of `deployment`'s epoch layer on `host`: the node the
    /// deployment's leader runs on.
    pub(crate) fn new(
        epochs: Arc<Epochs>,
        deployment: String,
        server_arguments: Vec<String>,
        link: EpochLink,
    ) -> Self {
        Self {
            epochs,
            deployment,
            server_arguments,
            link,
            launches: BTreeMap::new(),
            engines: BTreeMap::new(),
            builds: BTreeMap::new(),
            greeted: BTreeSet::new(),
            frontend: None,
            requests: String::new(),
            stopping: BTreeSet::new(),
            resuming: BTreeMap::new(),
            resuming_epoch: None,
            start: None,
            queued: VecDeque::new(),
            next_request: 1,
        }
    }

    /// Sends one request and waits for its answer.
    fn request(&mut self, kind: process::request::Kind) -> Result<process::Reply, String> {
        let id = self.next_request;
        self.next_request += 1;
        let (answer, reply) = std::sync::mpsc::channel();
        self.link
            .requests
            .send((
                Request {
                    id,
                    kind: Some(kind),
                },
                answer,
            ))
            .map_err(|_| "the daemon's service stopped".to_owned())?;
        reply
            .recv_timeout(REQUEST_TIMEOUT)
            .map_err(|_| format!("the daemon did not answer request {id}"))
    }

    /// The launch generation a peer runs, once a report of it named one.
    fn launch_of(&mut self, engine: u64, generation: u64) -> Option<u64> {
        if let Some(launch) = self.engines.get(&engine) {
            return Some(*launch);
        }
        self.launches.get(&generation)?;
        self.engines.insert(engine, generation);
        self.report_hello(engine, generation);
        Some(generation)
    }

    /// Tells the switch the build of the peer that now runs `generation`, once:
    /// the switch checks the build of every child a target starts.
    fn report_hello(&mut self, engine: u64, generation: u64) {
        if !self.greeted.insert(engine) {
            return;
        }
        let Some(build) = self.builds.get(&engine).cloned() else {
            return;
        };
        let Some(launch) = self.launches.get(&generation) else {
            return;
        };
        self.queued.push_back(Event::Hello {
            host: launch.host.clone(),
            generation,
            build,
        });
    }

    /// Writes the table row of `generation`, read-merge-write: the vector
    /// also holds the frontend's row and other epochs' rows, and only this
    /// node's launches are the adapter's to write.
    fn row(&self, generation: u64, state: ProcessState) {
        let Some(launch) = self.launches.get(&generation) else {
            return;
        };
        let mut processes = self.epochs.processes();
        let mut found = false;
        for process in &mut processes {
            let ours = process.role == Role::MainProcess
                && process.group == launch.group
                && process.rank == launch.rank
                && process.generation == generation;
            if ours {
                process.state = state.clone();
                process.pid = launch.pid;
                process.epoch = launch.epoch;
                found = true;
            }
        }
        if !found {
            processes.push(ManagedProcess {
                role: Role::MainProcess,
                group: launch.group.clone(),
                rank: launch.rank,
                pid: launch.pid,
                epoch: launch.epoch,
                build: String::new(),
                generation,
                state,
            });
        }
        self.epochs.set_processes(processes);
    }

    /// One report a peer sent.
    fn report(&mut self, engine: u64, report: Report) {
        use process::report::Kind;
        // The reports that name the launch generation of the peer that sent
        // them are how a connection is bound to the launch it runs.
        let launch = match &report.kind {
            Some(Kind::Parked(parked)) => Some(parked.generation),
            Some(Kind::Promoted(promoted)) => Some(promoted.generation),
            _ => None,
        };
        let launch = match launch {
            Some(generation) => self.launch_of(engine, generation),
            None => self.engines.get(&engine).copied(),
        };
        let Some(generation) = launch else {
            // A peer of no launch this epoch layer started: a worker of the
            // engine, or a connection from an earlier epoch. Nothing here
            // names a switch's node, so it is not the switch's.
            return;
        };
        let Some(launch) = self.launches.get(&generation) else {
            return;
        };
        // §7.3 step 4: a failure a main process reports during a program-layer
        // probation ends that probation in favour of the generation that served
        // before. The revert commands the processes and waits for their
        // answers, which arrive on the channel this thread reads, so it runs on
        // a thread of its own.
        if let Some(failure) = crate::epoch::Failure::of_report(&report) {
            let epochs = Arc::clone(&self.epochs);
            std::thread::Builder::new()
                .name("program-revert".into())
                .spawn(move || {
                    epochs.program_failure(failure);
                })
                .ok();
        }
        let host = launch.host.clone();
        if let Some(Kind::Drained(drained)) = &report.kind
            && let Some(launch) = self.launches.get_mut(&generation)
        {
            launch.epoch = Some(drained.epoch);
        }
        match &report.kind {
            Some(Kind::Parked(_)) => self.row(generation, ProcessState::Parked),
            Some(Kind::Promoted(_)) => self.row(generation, ProcessState::Serving),
            Some(Kind::Drained(_)) => self.row(generation, ProcessState::Parked),
            _ => {}
        }
        // A fresh generation of a resume serves as soon as it is promoted, and
        // its promotion is the epoch layer's own step.
        if self.resuming.contains_key(&generation)
            && matches!(report.kind, Some(Kind::Parked(_)))
            && let Some(peer) = self
                .engines
                .iter()
                .find(|(_, launch)| **launch == generation)
                .map(|(engine, _)| *engine)
        {
            if let Err(error) = self.link.commands.send(
                peer,
                command(process::command::Kind::Promote(process::Promote {})),
            ) {
                tracing::warn!(generation, %error, "promoting a restored epoch did not reach its main process");
            }
        }
        self.queued.push_back(Event::Report {
            host,
            generation,
            report,
        });
    }

    /// One supervised process exited.
    fn exited(&mut self, exit: Exit) {
        let generation = exit.generation;
        // A main process of the update in flight that exits ends the probation
        // (§7.3 step 4), on a thread of its own for the reason above.
        if let Some(program) = self.epochs.program_update()
            && program.update.state == crate::epoch::UpdateState::Probation
            && program
                .prepared
                .iter()
                .any(|prepared| prepared.generation == generation)
        {
            let epochs = Arc::clone(&self.epochs);
            let failure = crate::epoch::Failure {
                kind: "exited".to_owned(),
                reason: format!("a main process of the update exited ({})", exit.status),
                device: None,
            };
            std::thread::Builder::new()
                .name("program-revert".into())
                .spawn(move || {
                    epochs.program_failure(failure);
                })
                .ok();
        }
        let state = if self.stopping.remove(&generation) {
            ProcessState::Stopped
        } else {
            ProcessState::Failed(exit.status.clone())
        };
        self.row(generation, state);
        if let Some(launch) = self.launches.get_mut(&generation) {
            launch.pid = None;
        }
        self.resuming.remove(&generation);
        self.engines.retain(|_, launch| *launch != generation);
        self.queued.push_back(Event::Exit {
            host: exit.host,
            generation,
            code: exit.status.parse().ok(),
        });
    }

    /// Starts a fresh generation of the previous epoch for a restore whose
    /// processes are gone: claim its world, let it park, and promote it.
    fn fresh(&mut self, host: &str, epoch: u64, group: &str, rank: u64) {
        let scope = process::Scope {
            group: group.to_owned(),
            rank: rank as u32,
        };
        match self.claim(host, epoch, &[scope], &[]) {
            Ok(()) => {}
            Err(reason) => {
                tracing::warn!(%host, epoch, %reason, "claiming the previous epoch for a restore failed");
                return;
            }
        }
        let Some((executable, launch, environment)) = self.start.clone() else {
            return;
        };
        // The fresh process's own number is the spawn's to allocate and to
        // write into its launch file; the epoch waits for that number.
        self.resuming_epoch = Some(epoch);
        let start = Spawn {
            host: host.to_owned(),
            group: group.to_owned(),
            rank: rank as u32,
            generation: 0,
            executable,
            arguments: Vec::new(),
            environment,
            inherit: Vec::new(),
        };
        if let Err(reason) = self.spawn(start, &launch) {
            tracing::warn!(%host, epoch, %reason, "starting the previous epoch for a restore failed");
            self.resuming_epoch = None;
        }
    }
}

/// One command of `kind`.
fn command(kind: process::command::Kind) -> Command {
    Command { kind: Some(kind) }
}

impl Roster for ProcessRoster {
    fn claim(
        &mut self,
        host: &str,
        world: u64,
        scopes: &[process::Scope],
        retire: &[u64],
    ) -> Result<(), String> {
        let reply = self.request(process::request::Kind::Claim(process::Claim {
            host: host.to_owned(),
            deployment: self.deployment.clone(),
            scopes: scopes.to_vec(),
            server_arguments: self.server_arguments.clone(),
            world,
            retire: retire.to_vec(),
        }))?;
        if reply.error.is_empty() {
            Ok(())
        } else {
            Err(reply.error)
        }
    }

    fn spawn(
        &mut self,
        mut spawn: Spawn,
        launch: &crate::engine::launch::Launch,
    ) -> Result<crate::updates::switch::Spawned, String> {
        // The process's own identity, from the deployment's one allocator: a
        // launch file per process, so an epoch of two nodes never writes one
        // number for two children (§2.3, §7.4), and a fresh generation of a
        // restore never inherits the identity of the process it replaces.
        let generation = self
            .epochs
            .next_generation(&spawn.group, u64::from(spawn.rank));
        let mut launch = launch.clone();
        launch.engine = generation;
        let path =
            crate::engine::launch::write(&crate::options::runtime_directory(), generation, &launch)
                .map_err(|error| {
                    format!("writing the launch file of engine {generation}: {error}")
                })?;
        spawn.generation = generation;
        spawn.arguments = vec![
            "engine".to_owned(),
            "--launch".to_owned(),
            path.display().to_string(),
        ];
        self.start = Some((
            spawn.executable.clone(),
            launch.clone(),
            spawn.environment.clone(),
        ));
        self.launches.insert(
            generation,
            Launch {
                host: spawn.host.clone(),
                group: spawn.group.clone(),
                rank: u64::from(spawn.rank),
                epoch: None,
                pid: None,
            },
        );
        // A fresh generation of a restore: its epoch is now known under the
        // number this spawn allocated, so its `Parked` promotes it.
        if let Some(epoch) = self.resuming_epoch.take() {
            self.resuming.insert(generation, epoch);
        }
        let start = process::request::Kind::Spawn(spawn);
        let reply = self.request(start)?;
        if !reply.error.is_empty() {
            self.launches.remove(&generation);
            self.resuming.remove(&generation);
            return Err(reply.error);
        }
        let Some(process::reply::Kind::Spawned(spawned)) = reply.kind else {
            return Err("the daemon answered a spawn without a process id".to_owned());
        };
        if let Some(launch) = self.launches.get_mut(&generation) {
            launch.pid = Some(spawned.pid);
        }
        self.row(generation, ProcessState::Starting);
        // The generation the process was actually started at, which is what a
        // stop names and what a restore binds to.
        Ok(crate::updates::switch::Spawned {
            generation,
            pid: spawned.pid,
        })
    }

    fn stop(&mut self, _host: &str, group: &str, generation: Option<u64>) {
        if let Some(generation) = generation {
            self.stopping.insert(generation);
        } else {
            self.stopping.extend(
                self.launches
                    .iter()
                    .filter(|(_, launch)| launch.group == group)
                    .map(|(generation, _)| *generation),
            );
        }
        let reply = self.request(process::request::Kind::Stop(process::Stop {
            group: group.to_owned(),
            generation,
        }));
        if let Err(reason) = reply {
            tracing::warn!(%group, generation = ?generation, %reason, "the stop did not reach every node");
        }
    }

    fn command(&mut self, host: &str, generation: u64, command: Command) {
        let Some(launch) = self.launches.get(&generation) else {
            return;
        };
        if launch.host != host {
            return;
        }
        let Some(engine) = self
            .engines
            .iter()
            .find(|(_, launch)| **launch == generation)
            .map(|(engine, _)| *engine)
        else {
            tracing::warn!(%host, generation, "the main process of this generation is not connected");
            return;
        };
        if let Err(error) = self.link.commands.send(engine, command) {
            tracing::warn!(%host, generation, %error, "a command did not reach the main process");
        }
    }

    fn serve(&mut self, epoch: u64, requests: &str) {
        self.requests = requests.to_owned();
        let Some(frontend) = self.frontend else {
            tracing::warn!(
                epoch,
                "the frontend is not connected: nothing serves the epoch"
            );
            return;
        };
        if let Err(error) = self.link.commands.send(
            frontend,
            command(process::command::Kind::Serve(process::Serve {
                epoch,
                requests: requests.to_owned(),
            })),
        ) {
            tracing::warn!(epoch, %error, "serving the epoch did not reach the frontend");
        }
    }

    fn resume(&mut self, host: &str, generation: u64, reason: &str) {
        // The previous epoch's process of that launch is still alive — it
        // drained, not ended — so it serves again: §7.4 parks the epoch the
        // switchover drained.
        let alive = self
            .engines
            .iter()
            .find(|(_, launch)| **launch == generation)
            .map(|(engine, _)| *engine);
        if let Some(engine) = alive
            && let Some(launch) = self.launches.get(&generation)
            && launch.host == host
            && let Some(epoch) = launch.epoch
        {
            let _ = self.link.commands.send(
                engine,
                command(process::command::Kind::Serve(process::Serve {
                    epoch,
                    requests: self.requests.clone(),
                })),
            );
            tracing::info!(%host, generation, epoch, reason, "the previous epoch serves again");
            return;
        }
        // Its process is gone: a fresh generation of the epoch takes over,
        // above the target's (the switch records that generation as the
        // previous epoch's).
        let Some(launch) = self.launches.get(&generation) else {
            return;
        };
        let (group, rank, epoch) = (launch.group.clone(), launch.rank, launch.epoch);
        let Some(epoch) = epoch else {
            return;
        };
        self.fresh(host, epoch, &group, rank);
        let _ = reason;
    }

    fn events(&mut self) -> Vec<Event> {
        while let Ok((engine, greeting)) = self.link.greetings.try_recv() {
            if greeting.frontend {
                self.frontend = Some(engine);
            } else {
                // Its `Hello` reaches the switch with the report that names
                // the launch the peer runs.
                self.builds.insert(engine, greeting.build);
            }
        }
        let mut drained = Vec::new();
        while let Some((engine, report)) = self.link.commands.report(Duration::ZERO) {
            drained.push((engine, report));
        }
        for (engine, report) in drained {
            self.report(engine, report);
        }
        while let Ok(exit) = self.link.exits.try_recv() {
            self.exited(exit);
        }
        // One switch round reads them oldest first, each launch's `Hello`
        // before the report that named it.
        self.queued.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{Receiver, Sender};

    use super::*;
    use crate::epoch::{ProcessState, Role};
    use proto::process::report;

    /// A fake leader service: it answers every request the way a node's
    /// daemon would, and the test holds the peers' ends.
    struct Service {
        commands: Receiver<(u64, Command)>,
        reports: Sender<(u64, Report)>,
        exits: Sender<Exit>,
        greetings: Sender<(u64, Greeting)>,
    }

    fn service(info: Arc<Epochs>) -> (Service, EpochLink) {
        let (requests, incoming) = std::sync::mpsc::channel::<(Request, Sender<process::Reply>)>();
        let (_commands, sent) = std::sync::mpsc::channel();
        let (reports, _received) = std::sync::mpsc::channel();
        let (exits, exited) = std::sync::mpsc::channel();
        let (greetings, greeted) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            while let Ok((request, answer)) = incoming.recv() {
                let reply = match request.kind {
                    Some(process::request::Kind::Spawn(_)) => {
                        process::reply::Kind::Spawned(process::Spawned { pid: 4242 })
                    }
                    Some(process::request::Kind::Claim(claim)) => {
                        assert_eq!(claim.deployment, "deployment");
                        process::reply::Kind::Claimed(process::Claimed {
                            socket: "/run/storage.sock".to_owned(),
                            claim: Vec::new(),
                            tcp: String::new(),
                        })
                    }
                    Some(process::request::Kind::Stop(_)) => {
                        process::reply::Kind::Stopped(process::Stopped {})
                    }
                    _ => {
                        answer
                            .send(process::Reply {
                                id: request.id,
                                error: "unexpected request".to_owned(),
                                kind: None,
                            })
                            .expect("the epoch layer waits");
                        continue;
                    }
                };
                answer
                    .send(process::Reply {
                        id: request.id,
                        kind: Some(reply),
                        ..process::Reply::default()
                    })
                    .expect("the epoch layer waits");
            }
            drop(info);
        });
        let (processes, _commands_rx, _reports_tx) = ProcessCommands::pair();
        let link = EpochLink {
            requests,
            greetings: greeted,
            exits: exited,
            commands: processes,
        };
        let service = Service {
            commands: sent,
            reports,
            exits,
            greetings,
        };
        (service, link)
    }

    fn spawn(host: &str, generation: u64) -> Spawn {
        Spawn {
            host: host.to_owned(),
            group: "engine".to_owned(),
            rank: 0,
            generation,
            executable: "/opt/uniserve".to_owned(),
            arguments: vec!["engine".to_owned()],
            environment: Default::default(),
            inherit: Vec::new(),
        }
    }

    fn launch() -> crate::engine::launch::Launch {
        crate::engine::launch::Launch {
            build_config: "/run/build-config-1.json".into(),
            model: "/models/qwen3".into(),
            requests: "unix:/run/requests.sock".to_owned(),
            processes: "unix:/run/processes.sock".to_owned(),
            engine: 1,
            program_server: "/run/programs.sock".to_owned(),
            devices: vec![0],
            group: "engine".to_owned(),
            rank: 0,
            deployment: "deployment".to_owned(),
            world: 1,
            scheduler_node: true,
            peers: Vec::new(),
            watchdog_timeout_s: None,
            rollback_s: None,
            queue_depth: 4,
            world_size: 1,
            standby: false,
        }
    }

    fn roster(epochs: &Arc<Epochs>, link: EpochLink) -> ProcessRoster {
        ProcessRoster::new(
            Arc::clone(epochs),
            "deployment".to_owned(),
            Vec::new(),
            link,
        )
    }

    #[test]
    fn a_child_is_bound_by_the_launch_generation_it_reports_and_its_row_follows() {
        let epochs = Arc::new(Epochs::new());
        let (service, link) = service(Arc::clone(&epochs));
        let mut roster = roster(&epochs, link);
        // A row of another node's process is not the adapter's to replace.
        epochs.set_processes(vec![ManagedProcess {
            role: Role::Frontend,
            group: String::new(),
            rank: 0,
            pid: Some(1),
            epoch: None,
            build: "daemon".to_owned(),
            generation: 0,
            state: ProcessState::Serving,
        }]);

        let spawned = roster
            .spawn(spawn("n0", 7), &launch())
            .expect("the node starts it");
        assert_eq!(spawned.pid, 4242);
        let rows = epochs.processes();
        assert_eq!(rows.len(), 2, "the frontend's row survives the write");
        assert_eq!(rows[1].state, ProcessState::Starting);
        assert_eq!(rows[1].generation, 7);
        assert_eq!(rows[1].pid, Some(4242));

        // The child greets, then parks: the greeting becomes the switch's
        // `Hello` once the report names the launch.
        service
            .greetings
            .send((
                5,
                Greeting {
                    build: "2.0.0+new".to_owned(),
                    frontend: false,
                },
            ))
            .expect("the link reads greetings");
        service
            .reports
            .send((
                5,
                Report {
                    kind: Some(report::Kind::Parked(process::Parked { generation: 7 })),
                },
            ))
            .expect("the link reads reports");
        let events = roster.events();
        assert_eq!(
            events,
            vec![
                Event::Hello {
                    host: "n0".to_owned(),
                    generation: 7,
                    build: "2.0.0+new".to_owned(),
                },
                Event::Report {
                    host: "n0".to_owned(),
                    generation: 7,
                    report: Report {
                        kind: Some(report::Kind::Parked(process::Parked { generation: 7 })),
                    },
                },
            ]
        );
        assert_eq!(epochs.processes()[1].state, ProcessState::Parked);

        // The promotion by its launch generation reaches the peer that
        // reported it.
        roster.command(
            "n0",
            7,
            Command {
                kind: Some(process::command::Kind::Promote(process::Promote {})),
            },
        );
        let (engine, command) = service.commands.try_recv().expect("the command went down");
        assert_eq!(engine, 5);
        assert!(matches!(
            command.kind,
            Some(process::command::Kind::Promote(_))
        ));
    }

    #[test]
    fn a_drained_process_serves_again_for_a_resume() {
        let epochs = Arc::new(Epochs::new());
        let (service, link) = service(Arc::clone(&epochs));
        let mut roster = roster(&epochs, link);
        roster
            .spawn(spawn("n0", 7), &launch())
            .expect("the node starts it");
        service
            .greetings
            .send((
                5,
                Greeting {
                    build: "2.0.0+new".to_owned(),
                    frontend: false,
                },
            ))
            .unwrap();
        service
            .reports
            .send((
                5,
                Report {
                    kind: Some(report::Kind::Drained(process::Drained { epoch: 11 })),
                },
            ))
            .unwrap();
        let _ = roster.events();

        // The previous epoch's process drained and is still alive: the
        // restore serves it again instead of starting another.
        roster.resume("n0", 7, "revert");
        let (engine, command) = service.commands.try_recv().expect("a serve went down");
        assert_eq!(engine, 5);
        match command.kind {
            Some(process::command::Kind::Serve(serve)) => {
                assert_eq!(serve.epoch, 11);
            }
            other => panic!("a resume serves the drained epoch, not {other:?}"),
        }
    }

    #[test]
    fn a_stop_the_epoch_layer_asked_for_is_stopped_and_an_unasked_exit_failed() {
        let epochs = Arc::new(Epochs::new());
        let (service, link) = service(Arc::clone(&epochs));
        let mut roster = roster(&epochs, link);
        roster
            .spawn(spawn("n0", 7), &launch())
            .expect("the node starts it");
        roster
            .spawn(spawn("n0", 8), &launch())
            .expect("the node starts it");

        roster.stop("n0", "engine", Some(7));
        let exit = |generation: u64, status: &str| Exit {
            host: "n0".to_owned(),
            group: "engine".to_owned(),
            rank: 0,
            generation,
            status: status.to_owned(),
        };
        service.exits.send(exit(7, "exit status: 0")).unwrap();
        service.exits.send(exit(8, "signal: 11")).unwrap();
        let events = roster.events();
        assert!(events.contains(&Event::Exit {
            host: "n0".to_owned(),
            generation: 7,
            code: None,
        }));
        let rows = epochs.processes();
        assert_eq!(rows[0].state, ProcessState::Stopped);
        assert_eq!(rows[1].state, ProcessState::Failed("signal: 11".to_owned()));
    }

    #[test]
    fn the_frontend_is_served_the_epoch_the_switch_names() {
        let epochs = Arc::new(Epochs::new());
        let (service, link) = service(Arc::clone(&epochs));
        let mut roster = roster(&epochs, link);
        service
            .greetings
            .send((
                9,
                Greeting {
                    build: "2.0.0+new".to_owned(),
                    frontend: true,
                },
            ))
            .unwrap();
        let _ = roster.events();
        roster.serve(12, "unix:/run/requests.sock");
        let (engine, command) = service.commands.try_recv().expect("the frontend is told");
        assert_eq!(engine, 9);
        match command.kind {
            Some(process::command::Kind::Serve(serve)) => {
                assert_eq!(serve.epoch, 12);
                assert_eq!(serve.requests, "unix:/run/requests.sock");
            }
            other => panic!("the frontend serves an epoch, not {other:?}"),
        }
    }

    #[test]
    fn a_claim_names_the_deployment_the_world_and_the_retirement() {
        let epochs = Arc::new(Epochs::new());
        let (_service, link) = service(Arc::clone(&epochs));
        let mut roster = roster(&epochs, link);
        let scopes = vec![process::Scope {
            group: "engine".to_owned(),
            rank: 0,
        }];
        roster
            .claim("n0", 12, &scopes, &[11])
            .expect("the node claims the world");
    }
}
