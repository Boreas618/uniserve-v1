//! The epoch layer's [`Roster`] over the daemon's supervision.
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

use crate::epoch::{Epochs, ManagedProcess, ProcessState};
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
            let ours = process.group == launch.group
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
                // The vocabulary `Failure` names: a main process that ended
                // during probation (§7.3 step 4).
                kind: "process_exit".to_owned(),
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
        self.engines.retain(|_, launch| *launch != generation);
        self.queued.push_back(Event::Exit {
            host: exit.host,
            generation,
            code: exit.status.parse().ok(),
        });
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
        epoch: u64,
        scopes: &[process::Scope],
        retire: &[u64],
    ) -> Result<(), String> {
        let reply = self.request(process::request::Kind::Claim(process::Claim {
            host: host.to_owned(),
            deployment: self.deployment.clone(),
            scopes: scopes.to_vec(),
            server_arguments: self.server_arguments.clone(),
            epoch,
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
        launch: &proto::config::EngineArgs,
    ) -> Result<crate::updates::switch::Spawned, String> {
        // The launch generation is the caller's, allocated once from the
        // deployment's one counter ([`Epochs::next_generation`]): the switch
        // spawns an epoch's children at one number and commands them by it, so
        // taking a second number here would give one launch two identities —
        // the switch's and this node's — and the child's `Parked` would name a
        // generation no command reaches. A launch with no number is a caller
        // that allocated none, not a process to start (§2.3, §7.4).
        let generation = spawn.generation;
        if generation == 0 {
            return Err("a spawn states no launch generation".to_owned());
        }
        let mut launch = launch.clone();
        launch.engine = generation;
        // The node that starts the process writes its launch file (`Spawn::launch`).
        spawn.launch = serde_json::to_vec(&launch)
            .map_err(|error| format!("encoding the launch of engine {generation}: {error}"))?;
        spawn.arguments = vec!["engine".to_owned()];
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
        let start = process::request::Kind::Spawn(spawn);
        let reply = self.request(start)?;
        if !reply.error.is_empty() {
            self.launches.remove(&generation);
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

    fn adopt(&mut self, host: &str, group: &str, rank: u32, generation: u64) {
        self.launches.entry(generation).or_insert_with(|| Launch {
            host: host.to_owned(),
            group: group.to_owned(),
            rank: u64::from(rank),
            epoch: None,
            pid: None,
        });
        // A process of this deployment is numbered by the engine it states, which for one the
        // engine control started is its launch generation.
        self.engines.insert(generation, generation);
    }

    fn serve(&mut self, epoch: u64, requests: &str) {
        self.requests = requests.to_owned();
        // A frontend in this process follows the epochs' record, and one of its own hears the
        // command below: either way the record says what serves.
        self.epochs.announce_serve(epoch);
        let Some(frontend) = self.frontend else {
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
        // The previous epoch's process of that launch is alive: it drained, not
        // ended, so it serves again (§7.4). A process that is gone is the
        // switch's to report: its owner starts it again.
        let alive = self
            .engines
            .iter()
            .find(|(_, launch)| **launch == generation)
            .map(|(engine, _)| *engine);
        let Some(engine) = alive else {
            tracing::warn!(%host, generation, reason, "the previous epoch's process is gone");
            return;
        };
        let Some(epoch) = self
            .launches
            .get(&generation)
            .filter(|launch| launch.host == host)
            .and_then(|launch| launch.epoch)
        else {
            return;
        };
        if let Err(error) = self.link.commands.send(
            engine,
            command(process::command::Kind::Serve(process::Serve {
                epoch,
                requests: self.requests.clone(),
            })),
        ) {
            tracing::warn!(%host, generation, %error, "serving the previous epoch did not reach it");
        }
        tracing::info!(%host, generation, epoch, reason, "the previous epoch serves again");
    }

    fn keep_weights(
        &mut self,
        host: &str,
        epoch: u64,
        to_host: bool,
    ) -> Result<crate::updates::switch::KeptWeights, String> {
        let reply = self.request(process::request::Kind::KeepWeights(process::KeepWeights {
            host: host.to_owned(),
            deployment: self.deployment.clone(),
            epochs: vec![epoch],
            to_host,
            cache_keys: Vec::new(),
        }))?;
        if !reply.error.is_empty() {
            return Err(reply.error);
        }
        let Some(process::reply::Kind::WeightsKept(kept)) = reply.kind else {
            return Err("the daemon answered a retention without its report".to_owned());
        };
        Ok(crate::updates::switch::KeptWeights {
            released: kept.released,
            moved_bytes: kept.moved_bytes,
            refused: (!kept.refused.is_empty()).then_some(kept.refused),
        })
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
        // A caller that awaits the answers to the commands it sent (a program switch outside an
        // epoch change: the epoch's installation, a handover of its programs) reads the reports
        // itself; draining them here would take its answers away.
        let mut drained = Vec::new();
        while !self.link.commands.awaited()
            && let Some((engine, report)) = self.link.commands.report(Duration::ZERO)
        {
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
    use super::*;
    use crate::epoch::ProcessState;
    use crate::supervision::stand_in::{self, Refusals};
    use proto::process::{command, report};

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
            launch: Vec::new(),
        }
    }

    fn launch() -> proto::config::EngineArgs {
        proto::config::EngineArgs {
            engine_config: crate::options::ServerArgs::default()
                .engine_config()
                .expect("the configuration resolves"),
            predecessor: None,
            expert_order: Default::default(),
            host: "rank-0".to_owned(),
            requests: "unix:/run/requests.sock".to_owned(),
            processes: "unix:/run/processes.sock".to_owned(),
            engine: 1,
            program_server: "/run/programs.sock".to_owned(),
            devices: vec![0],
            group: "engine".to_owned(),
            rank: 0,
            epoch: 1,
            scheduler_node: true,
            world: None,
            standby_storage: crate::options::StandbyStorage::Resident,
            watchdog_timeout_s: None,
            rollback_s: None,
            queue_depth: 4,
            world_size: 1,
        }
    }

    #[test]
    fn a_child_is_bound_by_the_launch_generation_it_reports_and_its_row_follows() {
        let epochs = Arc::new(Epochs::new());
        let (nodes, mut roster) = stand_in::roster(&epochs, Refusals::default());
        // A row of another node's process is not the adapter's to replace.
        epochs.set_processes(vec![ManagedProcess {
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
        assert_eq!(spawned.pid, stand_in::pid(7));
        let rows = epochs.processes();
        assert_eq!(rows.len(), 2, "the frontend's row survives the write");
        assert_eq!(rows[1].state, ProcessState::Starting);
        assert_eq!(rows[1].generation, 7);
        assert_eq!(rows[1].pid, Some(stand_in::pid(7)));

        // The child greets, then parks: the greeting becomes the switch's
        // `Hello` once the report names the launch.
        nodes.greet(5, "2.0.0+new");
        nodes.report(5, report::Kind::Parked(process::Parked { generation: 7 }));
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
                kind: Some(command::Kind::Promote(process::Promote::default())),
            },
        );
        assert_eq!(
            nodes.sent(),
            vec![(5, command::Kind::Promote(process::Promote::default()))]
        );
    }

    #[test]
    fn a_drained_process_serves_again_for_a_resume() {
        let epochs = Arc::new(Epochs::new());
        let (nodes, mut roster) = stand_in::roster(&epochs, Refusals::default());
        roster
            .spawn(spawn("n0", 7), &launch())
            .expect("the node starts it");
        nodes.greet(5, "2.0.0+new");
        // The process parked before it served, which is the report that binds
        // its connection to the launch: `Drained` names no launch generation,
        // so a process drains under the binding it already has.
        nodes.report(5, report::Kind::Parked(process::Parked { generation: 7 }));
        nodes.report(
            5,
            report::Kind::Drained(process::Drained {
                epoch: 11,
                index: Vec::new(),
            }),
        );
        let _ = roster.events();

        // The previous epoch's process drained and is still alive: the
        // restore serves it again instead of starting another.
        roster.resume("n0", 7, "revert");
        assert_eq!(
            nodes.sent(),
            vec![(
                5,
                command::Kind::Serve(process::Serve {
                    epoch: 11,
                    requests: String::new(),
                })
            )]
        );
    }

    #[test]
    fn a_stop_the_epoch_layer_asked_for_is_stopped_and_an_unasked_exit_failed() {
        let epochs = Arc::new(Epochs::new());
        let (nodes, mut roster) = stand_in::roster(&epochs, Refusals::default());
        roster
            .spawn(spawn("n0", 7), &launch())
            .expect("the node starts it");
        roster
            .spawn(spawn("n0", 8), &launch())
            .expect("the node starts it");

        roster.stop("n0", "engine", Some(7));
        assert_eq!(nodes.stops(), vec![Some(7)]);
        nodes.exit("n0", 7, "exit status: 0");
        nodes.exit("n0", 8, "signal: 11");
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
        let (nodes, mut roster) = stand_in::roster(&epochs, Refusals::default());
        nodes.greet_frontend(9);
        let _ = roster.events();
        roster.serve(12, "unix:/run/requests.sock");
        assert_eq!(
            nodes.sent(),
            vec![(
                9,
                command::Kind::Serve(process::Serve {
                    epoch: 12,
                    requests: "unix:/run/requests.sock".to_owned(),
                })
            )]
        );
        assert_eq!(*epochs.served().borrow(), 12);
    }

    #[test]
    fn a_claim_names_the_deployment_the_epoch_and_the_retirement() {
        let epochs = Arc::new(Epochs::new());
        let (nodes, mut roster) = stand_in::roster(&epochs, Refusals::default());
        let scopes = vec![process::Scope {
            group: "engine".to_owned(),
            rank: 0,
        }];
        roster
            .claim("n0", 12, &scopes, &[11])
            .expect("the node claims the epoch");
        let claims = nodes.claims();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].host, "n0");
        assert_eq!(claims[0].deployment, "deployment");
        assert_eq!(claims[0].epoch, 12);
        assert_eq!(claims[0].scopes, scopes);
        assert_eq!(claims[0].retire, vec![11]);
    }
}
