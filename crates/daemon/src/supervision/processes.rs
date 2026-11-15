//! The deployment's engine processes as the leader's daemon commands them.
//!
//! [`Processes`] owns the deployment's engines for the daemon's life: it starts
//! and stops them through the leader service, commands the engine of one launch
//! generation, and turns what the engines report into the update's [`Event`]s.
//! It is also the writer of the node's process table (`GET /admin/config`):
//! every launch is one row, from `Starting` at its spawn to `Parked`,
//! `Serving`, `Stopped` or `Failed`.
//!
//! A launch generation is the identity of one process: the daemon launches an
//! engine under it, the engine states it in the hello of its process link,
//! and the leader's service files that link under it, so a command for a
//! launch generation reaches its engine and a report names the launch it
//! comes from. The generation of modules travels inside each
//! [`process::Command`].

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use proto::admin::{ManagedProcess, ProcessState};
use proto::process::{self, Command, Exit, Report, Request, Spawn};

use crate::epoch::Epochs;
use crate::supervision::{ProcessCommands, ToService};

/// The update's ends of the leader service: requests down, and what the
/// service sees up.
pub(crate) struct ProcessLink {
    /// Requests to the deployment's daemons, each with where its reply goes.
    pub(crate) requests: ToService<(Request, Sender<process::Reply>)>,
    /// Every engine whose process link greeted the leader, by its launch generation, with the
    /// build its hello states.
    pub(crate) greetings: Receiver<(u64, String)>,
    /// Every supervised process that exited.
    pub(crate) exits: Receiver<Exit>,
    /// Commands to a connected peer and the reports it sends.
    pub(crate) commands: ProcessCommands,
}

/// What the deployment's engines tell an update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Event {
    /// The engine of launch generation `generation` on `host` connected,
    /// saying what build it runs.
    Hello {
        host: String,
        generation: u64,
        build: String,
    },
    /// The engine of launch generation `generation` on `host` reported.
    Report {
        host: String,
        generation: u64,
        report: Report,
    },
    /// The engine of launch generation `generation` on `host` exited.
    Exit {
        host: String,
        generation: u64,
        code: Option<i32>,
    },
}

/// One engine process the daemon commands.
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

/// The engine processes of one deployment: their launches, and the link they are commanded
/// through.
pub(crate) struct Processes {
    epochs: Arc<Epochs>,
    /// The deployment's name, which every claim states.
    deployment: String,
    link: ProcessLink,
    /// The launches by launch generation. The generation is unique across the
    /// deployment's engine group, so a number names one launch.
    launches: BTreeMap<u64, Launch>,
    /// Launches the daemon stopped: their exit is not a failure.
    stopping: BTreeSet<u64>,
    /// Events raised here, ahead of the ones still on the link.
    queued: VecDeque<Event>,
    next_request: u64,
}

impl Processes {
    /// The engine processes of `deployment`, commanded through `link`.
    pub(crate) fn new(epochs: Arc<Epochs>, deployment: String, link: ProcessLink) -> Self {
        Self {
            epochs,
            deployment,
            link,
            launches: BTreeMap::new(),
            stopping: BTreeSet::new(),
            queued: VecDeque::new(),
            next_request: 1,
        }
    }

    /// Sends one request and waits for its reply. The service replies to every request: at once
    /// for this node, once a member replied for another node, and with the member's loss when
    /// that node's daemon is lost; a service that ended drops the reply's sender.
    fn request(&mut self, kind: process::request::Kind) -> Result<process::Reply, String> {
        let id = self.next_request;
        self.next_request += 1;
        let (reply, replied) = std::sync::mpsc::channel();
        self.link
            .requests
            .send((
                Request {
                    id,
                    kind: Some(kind),
                },
                reply,
            ))
            .map_err(|_| "the daemon's service stopped".to_owned())?;
        replied
            .recv()
            .map_err(|_| format!("the daemon's service ended before it replied to request {id}"))
    }

    /// Raises the `Hello` of the engine of launch generation `generation`, which greeted with
    /// `build`: an update checks the build of every engine it starts. An engine of no launch
    /// known here (one the daemon neither started nor adopted) raises nothing.
    fn greeted(&mut self, generation: u64, build: String) {
        let Some(launch) = self.launches.get(&generation) else {
            return;
        };
        self.queued.push_back(Event::Hello {
            host: launch.host.clone(),
            generation,
            build,
        });
    }

    /// Writes the table row of `generation`, read-merge-write: the table also
    /// holds the rows the leader service writes, and only the launches known
    /// here are written here.
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

    /// One report the engine of launch generation `generation` sent on its process link.
    fn report(&mut self, generation: u64, report: Report) {
        use process::report::Kind;

        let Some(launch) = self.launches.get_mut(&generation) else {
            // An engine of no launch known here: one the daemon neither started nor adopted.
            // Nothing names a node for it.
            return;
        };
        let host = launch.host.clone();
        if let Some(Kind::Drained(drained)) = &report.kind {
            launch.epoch = Some(drained.epoch);
        }

        match &report.kind {
            Some(Kind::Parked(_) | Kind::Drained(_)) => self.row(generation, ProcessState::Parked),
            Some(Kind::Serving(_)) => self.row(generation, ProcessState::Serving),
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
        let state = if self.stopping.remove(&generation) {
            ProcessState::Stopped
        } else {
            ProcessState::Failed(exit.status.clone())
        };
        self.row(generation, state);
        if let Some(launch) = self.launches.get_mut(&generation) {
            launch.pid = None;
        }
        self.queued.push_back(Event::Exit {
            host: exit.host,
            generation,
            code: exit.status.parse().ok(),
        });
    }

    /// Claims `epoch`'s storage on `host` for `scopes`, retiring `retire`'s
    /// epochs, and holds the claim when it returns `Ok`: the claim is held
    /// before an engine of the epoch maps storage, so it is synchronous and
    /// has no matching report. `Err` says the node refused and holds nothing.
    pub(crate) fn claim(
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
            // The node's daemon states the deployment's Storage Server settings.
            storage: None,
            epoch,
            retire: retire.to_vec(),
        }))?;
        if reply.error.is_empty() {
            Ok(())
        } else {
            Err(reply.error)
        }
    }

    /// Starts the engine `spawn` names from `launch`, and returns its process
    /// id.
    ///
    /// The launch generation is the caller's, allocated once from the
    /// deployment's one allocator (`EngineControl::launch_number`): an engine is
    /// started at one number and commanded by it, so a second number taken
    /// here would give one launch two identities.
    pub(crate) fn spawn(
        &mut self,
        mut spawn: Spawn,
        launch: &proto::config::EngineArgs,
    ) -> Result<u32, String> {
        let generation = spawn.generation;
        if generation == 0 {
            return Err("a spawn states no launch generation".to_owned());
        }
        let mut launch = launch.clone();
        launch.engine = generation;
        // The node that starts the process writes its launch file (`Spawn::launch`) and runs the
        // engine package's executable with `--launch` naming it, which is the whole command line
        // of a main process: the spawn's own arguments are left as the caller stated them.
        spawn.launch = serde_json::to_vec(&launch)
            .map_err(|error| format!("encoding the launch of engine {generation}: {error}"))?;
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

        let reply = self.request(process::request::Kind::Spawn(spawn))?;
        if !reply.error.is_empty() {
            self.launches.remove(&generation);
            return Err(reply.error);
        }
        let Some(process::reply::Kind::Spawned(spawned)) = reply.kind else {
            self.launches.remove(&generation);
            return Err("the daemon replied to a spawn without a process id".to_owned());
        };
        if let Some(launch) = self.launches.get_mut(&generation) {
            launch.pid = Some(spawned.pid);
        }
        self.row(generation, ProcessState::Starting);
        Ok(spawned.pid)
    }

    /// Stops one launch generation of `group` on every node; every generation
    /// when `generation` is `None`.
    pub(crate) fn stop(&mut self, group: &str, generation: Option<u64>) {
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

    /// Commands the engine of launch generation `generation` on `host`: the leader's service
    /// carries it down the process link that engine opened under that number.
    pub(crate) fn command(&mut self, host: &str, generation: u64, command: Command) {
        let Some(launch) = self.launches.get(&generation) else {
            return;
        };
        if launch.host != host {
            return;
        }
        if let Err(error) = self.link.commands.send(generation, command) {
            tracing::warn!(%host, generation, %error, "a command did not reach the engine");
        }
    }

    /// Records the engine of `generation` on `host`, which the daemon's engine control started
    /// rather than this updater, so it is commanded and reported like the engines started here.
    pub(crate) fn adopt(&mut self, host: &str, group: &str, rank: u32, generation: u64) {
        self.launches.entry(generation).or_insert_with(|| Launch {
            host: host.to_owned(),
            group: group.to_owned(),
            rank: u64::from(rank),
            epoch: None,
            pid: None,
        });
    }

    /// States that `epoch` serves from now on: the record every request path of this daemon
    /// follows.
    pub(crate) fn serve(&mut self, epoch: u64) {
        self.epochs.announce_serve(epoch);
    }

    /// Retains `epoch`'s weights on `host` once its processes exited, and
    /// releases the rest of its exclusive storage; with `to_host` the weights
    /// it alone retains move to host storage. Returns what the node did.
    pub(crate) fn keep_weights(
        &mut self,
        host: &str,
        epoch: u64,
        to_host: bool,
    ) -> Result<process::WeightsKept, String> {
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
        match reply.kind {
            Some(process::reply::Kind::WeightsKept(kept)) => Ok(kept),
            _ => Err("the daemon replied to a retention without its report".to_owned()),
        }
    }

    /// Takes the events that arrived since the last call, oldest first, each
    /// launch's `Hello` before the report that named it.
    pub(crate) fn events(&mut self) -> Vec<Event> {
        while let Ok((generation, build)) = self.link.greetings.try_recv() {
            self.greeted(generation, build);
        }
        while let Some((generation, report)) = self.link.commands.report(Duration::ZERO) {
            self.report(generation, report);
        }
        while let Ok(exit) = self.link.exits.try_recv() {
            self.exited(exit);
        }
        self.queued.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::Engines;
    use proto::admin::ProcessState;
    use proto::process::{command, report};
    use testkit::process::Refusals;

    const BUILD: &str = "2.0.0+new";

    /// The processes of a deployment whose engines and nodes are fakes.
    fn processes() -> (Arc<Epochs>, Engines, Processes) {
        let epochs = Arc::new(Epochs::default());
        let (engines, link) = Engines::link(Refusals::default());
        let processes = Processes::new(Arc::clone(&epochs), "deployment".to_owned(), link);
        (epochs, engines, processes)
    }

    fn spawn(host: &str, generation: u64) -> Spawn {
        Spawn {
            host: host.to_owned(),
            group: "engine".to_owned(),
            rank: 0,
            generation,
            inherit: Vec::new(),
            launch: Vec::new(),
        }
    }

    /// A launch that names the engine it succeeds, so the engine parks.
    fn launch() -> proto::config::EngineArgs {
        let options = crate::options::ServerArgs::default();
        let mut launch = proto::config::EngineArgs::new(
            options.engine_config().expect("the configuration resolves"),
            crate::epoch::modules_of(&options),
        );
        launch.predecessor = Some(proto::request::wire::Predecessor { epoch: 1 });
        launch
    }

    /// An engine is the launch generation it was launched under: its `Hello`
    /// is raised before its first report, its row follows its lifecycle, and a
    /// command by launch generation reaches it.
    #[test]
    fn an_engine_is_the_launch_generation_it_was_launched_under() {
        let (epochs, mut engines, mut processes) = processes();
        // A row another writer keeps is not replaced.
        epochs.set_processes(vec![ManagedProcess {
            group: String::new(),
            rank: 0,
            pid: Some(1),
            epoch: None,
            build: "daemon".to_owned(),
            generation: 0,
            state: ProcessState::Serving,
        }]);

        let pid = processes
            .spawn(spawn("n0", 7), &launch())
            .expect("the node starts it");
        assert_eq!(pid, testkit::process::pid(7));
        let rows = epochs.processes();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].state, ProcessState::Starting);
        assert_eq!((rows[1].generation, rows[1].pid), (7, Some(pid)));

        assert!(engines.start_spawned(BUILD));
        let events = processes.events();
        assert_eq!(
            events[0],
            Event::Hello {
                host: "n0".to_owned(),
                generation: 7,
                build: BUILD.to_owned(),
            }
        );
        assert!(matches!(
            &events[1],
            Event::Report { generation: 7, report, .. }
                if matches!(report.kind, Some(report::Kind::Parked(_)))
        ));
        assert_eq!(epochs.processes()[1].state, ProcessState::Parked);

        let promote = Command {
            kind: Some(command::Kind::Promote(process::Promote::default())),
        };
        processes.command("n0", 7, promote.clone());
        // Another host's engine of that number does not exist.
        processes.command("n1", 7, promote);
        assert!(engines.deliver());
        assert_eq!(engines.received(7).len(), 1);
        processes.events();
        assert_eq!(epochs.processes()[1].state, ProcessState::Serving);
    }

    /// An exit the daemon asked for is a stop; one it did not ask for is a
    /// failure with the status the node read.
    #[test]
    fn a_stop_is_stopped_and_an_exit_nobody_asked_for_failed() {
        let (epochs, engines, mut processes) = processes();
        for generation in [7, 8] {
            processes
                .spawn(spawn("n0", generation), &launch())
                .expect("the node starts it");
        }
        processes.stop("engine", Some(7));
        assert_eq!(engines.stops(), vec![Some(7)]);
        engines.exit("n0", 7, "exit status: 0");
        engines.exit("n0", 8, "signal: 11");
        let events = processes.events();
        assert!(events.contains(&Event::Exit {
            host: "n0".to_owned(),
            generation: 7,
            code: None,
        }));
        let rows = epochs.processes();
        assert_eq!(rows[0].state, ProcessState::Stopped);
        assert_eq!(rows[1].state, ProcessState::Failed("signal: 11".to_owned()));
    }

    /// A claim names the deployment, the epoch, its scopes and what it
    /// retires.
    #[test]
    fn a_claim_names_the_deployment_the_epoch_and_what_it_retires() {
        let (_, engines, mut processes) = processes();
        let scopes = vec![process::Scope {
            group: "engine".to_owned(),
            rank: 0,
        }];
        processes
            .claim("n0", 12, &scopes, &[11])
            .expect("the node claims the epoch");
        let claims = engines.claims();
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].host, "n0");
        assert_eq!(claims[0].deployment, "deployment");
        assert_eq!(claims[0].epoch, 12);
        assert_eq!(claims[0].scopes, scopes);
        assert_eq!(claims[0].retire, vec![11]);
    }
}
