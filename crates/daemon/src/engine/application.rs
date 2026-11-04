//! The application process of a node: the frontend the daemon starts, gates and
//! reclaims (`specs/worker.md` §2.1, §2.3, §7.4).
//!
//! A node runs two kinds of process under its daemon: the main processes, which
//! execute (§2.5), and the frontend, which serves the application — the HTTP
//! and OpenAI API, input and output processing, the facts of the requests in
//! flight and the metrics of the serving address. This module is the daemon's
//! side of the second kind, and only that side:
//!
//! - [`Application::start`] spawns `uniserve frontend` in the discipline the
//!   main processes use: an executable, its arguments and its environment
//!   handed to the node's service, which starts it and answers its process id.
//! - [`Application::greeted`] takes the peer its `Hello` came from, which is the
//!   address every command to it goes to.
//! - [`Application::pause`] is the frontend's admission gate, the frontend half
//!   of `/admin/pause_generation`: no new request starts, the ones in flight go
//!   on, and the gate is a separate knob from the engine that holds them.
//! - [`Application::adopt`] keeps the process across a hot restart. A process
//!   is not a connection: the successor takes the same frontend over rather
//!   than starting a second one, exactly as it keeps the engines (§7.5).
//! - [`Application::stop`] ends it, and it is the last thing a shutdown does,
//!   so no in-flight response is cut by the daemon's own exit.
//!
//! Which epoch the frontend serves is not decided here. The epoch layer sends
//! `Serve{epoch, requests}` to the frontend the greeting named
//! (`ProcessRoster::serve`), because the switch is what knows an epoch is
//! ready; a second account of the same order from this module would race it on
//! the same peer. Commands go only downward (§2.3): a command that cannot be
//! sent is an error and not a retry, and the caller decides what an epoch may
//! serve.

use std::collections::HashMap;
use std::sync::mpsc::Sender;

use proto::process;

use crate::supervision::ProcessCommands;

/// How a node's service is asked to start and stop a process: the request
/// channel every launch goes through, and where its answer goes.
type Request = (
    process::Request,
    Sender<process::Reply>,
);

/// The application process of the node: what the daemon needs to start it,
/// gate it and end it.
pub(crate) struct Application {
    /// The node's service, which starts and stops processes.
    requests: Sender<Request>,
    /// Commands to the connected peers, the frontend among them.
    commands: ProcessCommands,
    /// The executable and the arguments the frontend is started with: the
    /// deployment's own, so a frontend reads the configuration the daemon
    /// resolved and not a command line of its own (§7.5, §7.7).
    executable: String,
    arguments: Vec<String>,
    environment: HashMap<String, String>,
    /// The launch generation of this process, unique among the node's launches.
    generation: u64,
    /// The process id the service answered, once it started.
    pid: Option<u32>,
    /// The peer the frontend's `Hello` came from, once it greeted: the address
    /// its commands go to.
    peer: Option<u64>,
    /// Whether new requests wait at the frontend's gate.
    paused: bool,
}

/// The group the frontend is started under, so that a stop names it and not a
/// worker group of a deployment.
pub(crate) const FRONTEND: &str = "frontend";

impl Application {
    /// The arguments a frontend is started with, from the ones the daemon was
    /// started with: the same file, flags and `--set` assignments, with the
    /// `serve` subcommand replaced by `frontend` (§7.5, §7.7).
    ///
    /// The deployment has one parser, and the frontend command resolves the
    /// same arguments `serve` does, so what the daemon resolved is what the
    /// frontend resolves; nothing is re-stated as a second command line.
    pub(crate) fn arguments(argv: &[String]) -> Vec<String> {
        let mut arguments = argv.to_vec();
        if let Some(subcommand) = arguments.first_mut()
            && subcommand == "serve"
        {
            *subcommand = "frontend".to_owned();
        }
        arguments
    }

    /// The frontend of the node running `executable` with `arguments` and
    /// `environment`, started through `requests` and commanded through
    /// `commands`.
    ///
    /// `generation` names this launch in the node's service; a hot restart that
    /// adopts the process keeps it, so it survives the daemon that started it.
    pub(crate) fn new(
        requests: Sender<Request>,
        commands: ProcessCommands,
        executable: impl Into<String>,
        arguments: Vec<String>,
        environment: HashMap<String, String>,
        generation: u64,
    ) -> Self {
        Self {
            requests,
            commands,
            executable: executable.into(),
            arguments,
            environment,
            generation,
            pid: None,
            peer: None,
            paused: false,
        }
    }

    /// Starts the frontend and answers the process id the node's service
    /// reported.
    pub(crate) fn start(&mut self) -> anyhow::Result<u32> {
        let pid = self.request(process::request::Kind::Spawn(process::Spawn {
            host: String::new(),
            group: FRONTEND.to_owned(),
            rank: 0,
            generation: self.generation,
            executable: self.executable.clone(),
            arguments: self.arguments.clone(),
            environment: self.environment.clone(),
            inherit: Vec::new(),
        }))?;
        self.pid = Some(pid);
        tracing::info!(pid, generation = self.generation, "the frontend started");
        Ok(pid)
    }

    /// Takes the peer the frontend greeted from: every command to it goes
    /// there, and until it greets there is nothing to command.
    pub(crate) fn greeted(&mut self, peer: u64) {
        self.peer = Some(peer);
    }

    /// Holds new requests at the frontend's gate (`paused`) or releases them.
    pub(crate) fn pause(&mut self, paused: bool) -> anyhow::Result<()> {
        let peer = self.peer.ok_or_else(|| {
            anyhow::anyhow!("the frontend has not greeted: no peer to gate")
        })?;
        self.commands
            .send(
                peer,
                process::Command {
                    kind: Some(process::command::Kind::Pause(process::Pause {
                        paused,
                    })),
                },
            )
            .map_err(|error| anyhow::anyhow!("gating the frontend failed: {error}"))?;
        self.paused = paused;
        Ok(())
    }

    /// The same process under a successor daemon: a hot restart keeps the
    /// frontend rather than starting a second one, and the peer it greeted with
    /// is the successor's own connection, so that is left to be greeted again.
    pub(crate) fn adopt(&mut self, pid: u32, generation: u64) {
        self.pid = Some(pid);
        self.generation = generation;
        self.peer = None;
        self.paused = false;
    }

    /// Ends the frontend, at the end of a shutdown. A second call has nothing
    /// left to end and sends nothing.
    pub(crate) fn stop(&mut self) -> anyhow::Result<()> {
        if self.pid.take().is_none() {
            return Ok(());
        }
        self.peer = None;
        self.paused = false;
        self.request(process::request::Kind::Stop(process::Stop {
            group: FRONTEND.to_owned(),
            generation: Some(self.generation),
        }))?;
        Ok(())
    }

    /// The process id of the frontend, while it runs.
    pub(crate) fn pid(&self) -> Option<u32> {
        self.pid
    }

    /// Whether new requests wait at the frontend's gate.
    pub(crate) fn paused(&self) -> bool {
        self.paused
    }

    /// Sends one request to the node's service and answers the process id it
    /// reported.
    fn request(&mut self, kind: process::request::Kind) -> anyhow::Result<u32> {
        let id = self.generation;
        let (answer, replied) = std::sync::mpsc::channel();
        self.requests
            .send((
                process::Request {
                    id,
                    kind: Some(kind),
                },
                answer,
            ))
            .map_err(|_| anyhow::anyhow!("the node's service stopped"))?;
        // The service answers a launch from its own control path, so the wait
        // is a spawn, or a node that is wedged; it never carries request
        // traffic, and this is called from the daemon's start and shutdown and
        // not from a request handler.
        let reply = replied
            .recv_timeout(std::time::Duration::from_secs(120))
            .map_err(|_| anyhow::anyhow!("the node's service did not answer"))?;
        if !reply.error.is_empty() {
            anyhow::bail!("{}", reply.error);
        }
        match reply.kind {
            Some(process::reply::Kind::Spawned(spawned)) => Ok(spawned.pid),
            Some(process::reply::Kind::Stopped(_)) => Ok(0),
            other => anyhow::bail!("the service answered a launch with {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The application under test, and the commands the frontend received.
    struct Fixture {
        application: Application,
        commands: std::sync::mpsc::Receiver<(u64, process::Command)>,
    }

    /// The application of a node whose service answers every request: process
    /// id 41 for a launch, and the empty answer for a stop.
    fn fixture() -> Fixture {
        let (requests, launched) = std::sync::mpsc::channel::<Request>();
        std::thread::spawn(move || {
            for (request, answer) in launched {
                let kind = match request.kind {
                    Some(process::request::Kind::Spawn(_)) => {
                        process::reply::Kind::Spawned(process::Spawned { pid: 41 })
                    }
                    _ => process::reply::Kind::Stopped(process::Stopped {}),
                };
                let _ = answer.send(process::Reply {
                    id: request.id,
                    kind: Some(kind),
                    ..Default::default()
                });
            }
        });
        let (commands, received, _reports) = ProcessCommands::channel();
        Fixture {
            application: Application::new(
                requests,
                commands,
                "uniserve",
                vec!["frontend".to_owned(), "--port".to_owned(), "8000".to_owned()],
                HashMap::new(),
                7,
            ),
            commands: received,
        }
    }

    /// The next command the frontend received, with the peer it was sent to.
    fn command(
        received: &std::sync::mpsc::Receiver<(u64, process::Command)>,
    ) -> (u64, process::Command) {
        received
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("the frontend received a command")
    }

    #[test]
    fn the_launch_hands_the_node_its_process_and_keeps_its_pid() {
        let mut fixture = fixture();
        assert_eq!(fixture.application.start().expect("the launch"), 41);
        assert_eq!(fixture.application.pid(), Some(41));
        assert!(!fixture.application.paused());
    }

    #[test]
    fn the_frontend_is_started_with_the_deployments_own_arguments() {
        let argv: Vec<String> = [
            "serve",
            "/models/qwen3",
            "--port",
            "8000",
            "--set",
            "kv_cache_config.host_gib=32",
        ]
        .iter()
        .map(|argument| (*argument).to_owned())
        .collect();
        assert_eq!(
            Application::arguments(&argv),
            [
                "frontend",
                "/models/qwen3",
                "--port",
                "8000",
                "--set",
                "kv_cache_config.host_gib=32",
            ]
        );
        // Whatever the daemon was started as is handed on unchanged when it is
        // not the `serve` subcommand; the frontend's own parser decides.
        let other: Vec<String> = ["frontend", "--port", "8001"]
            .iter()
            .map(|argument| (*argument).to_owned())
            .collect();
        assert_eq!(Application::arguments(&other), other);
    }

    #[test]
    fn nothing_is_commanded_before_the_frontend_greets() {
        let mut fixture = fixture();
        let _ = fixture.application.start().expect("the launch");
        assert!(
            fixture.application.pause(true).is_err(),
            "a gate command to no peer is refused"
        );
    }

    #[test]
    fn the_gate_holds_admissions_in_the_frontends_words() {
        let mut fixture = fixture();
        let _ = fixture.application.start().expect("the launch");
        fixture.application.greeted(9);

        fixture
            .application
            .pause(true)
            .expect("the admissions are held");
        let (peer, held) = command(&fixture.commands);
        assert_eq!(peer, 9, "the command goes to the frontend's peer");
        match held.kind {
            Some(process::command::Kind::Pause(pause)) => assert!(pause.paused),
            other => panic!("expected Pause, got {other:?}"),
        }
        assert!(fixture.application.paused());

        fixture
            .application
            .pause(false)
            .expect("the admissions are released");
        match command(&fixture.commands).1.kind {
            Some(process::command::Kind::Pause(pause)) => assert!(!pause.paused),
            other => panic!("expected Pause, got {other:?}"),
        }
        assert!(!fixture.application.paused());
    }

    #[test]
    fn a_hot_restart_keeps_the_process_and_a_shutdown_ends_it_once() {
        let mut fixture = fixture();
        let _ = fixture.application.start().expect("the launch");
        fixture.application.greeted(9);
        fixture.application.pause(true).expect("the gate closed");

        // The successor takes the same process over: no second launch, the same
        // pid, and the peer is the successor's own connection, which greets
        // afresh.
        fixture.application.adopt(41, 8);
        assert_eq!(fixture.application.pid(), Some(41));
        assert!(!fixture.application.paused());
        assert!(
            fixture.application.pause(false).is_err(),
            "the adopted process has not greeted on this connection yet"
        );

        // The stop is a request to the node's service, which the fixture
        // answers; the process id it kept is what says it ended.
        fixture.application.stop().expect("the frontend stops");
        assert_eq!(fixture.application.pid(), None);
        // A second stop ends nothing twice, and sends nothing.
        fixture.application.stop().expect("no process is left");
    }
}
