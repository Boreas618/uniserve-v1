//! Starting the deployment's engine process and connecting to it.
//!
//! The leader daemon starts each engine process as a child of its own node's
//! supervision, through the process interface like any process of the
//! deployment: [`EngineControl`] connects to the daemon's process socket and
//! asks it to spawn `uniserve engine`, which connects back to the daemon's
//! request socket and says hello. The daemon answers with the engine's
//! specification and waits until the engine reports that it serves; the
//! engine's process lives for as long as its control connection does.

use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context as _;
use proto::Connection;
use proto::process::{self, downstream, event, hello, reply, request, upstream};
use proto::request::wire;

use crate::engine::client::{EngineClient, EngineState, StatsSink};

/// The worker group a daemon supervises an engine process under.
const ENGINE_GROUP: &str = "engine";

/// How long an engine process takes to connect back to the request socket
/// once the daemon started it.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// How long the daemon takes to answer the control connection.
const DAEMON_TIMEOUT: Duration = Duration::from_secs(90);

/// How often a wait rechecks the engine process while it starts.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Starts the engine processes of one deployment.
pub struct EngineControl {
    /// The request socket engine processes connect back to.
    requests: UnixListener,
    request_address: String,
    socket: PathBuf,
    /// The daemon's process socket.
    processes: String,
    /// This node's host identity, where the engine process runs.
    host: String,
    /// The deployment the engines serve.
    deployment: String,
    next_engine: AtomicU64,
}

impl EngineControl {
    /// Binds the request socket at `socket` for the engines of `deployment`,
    /// which run on host `host` through the daemon whose
    /// process socket is `processes`.
    ///
    /// # Errors
    ///
    /// Fails when the socket cannot be bound.
    pub fn new(
        socket: &Path,
        processes: String,
        host: String,
        deployment: String,
    ) -> anyhow::Result<Self> {
        // A socket a stopped daemon left behind is replaced.
        let _ = std::fs::remove_file(socket);
        let requests = UnixListener::bind(socket)
            .with_context(|| format!("binding the request socket {}", socket.display()))?;
        requests.set_nonblocking(true)?;
        Ok(Self {
            requests,
            request_address: format!("unix:{}", socket.display()),
            socket: socket.to_path_buf(),
            processes,
            host,
            deployment,
            next_engine: AtomicU64::new(1),
        })
    }

    /// Starts an engine process that serves `spec` and returns its client
    /// once it reports that it serves, or, for a spec that stands by for the
    /// running engine (`EngineSpec::standby`), once it reports that it
    /// stands by; `stats` receives its schedulers' statistics. The process
    /// runs the executable of the spec's build (`workers.executable`).
    /// Blocks for the engine's whole start, its workers' loading included.
    ///
    /// # Errors
    ///
    /// Fails when the daemon cannot start the process, the process exits or
    /// does not connect back, its build speaks another major version, or it
    /// reports that its start failed.
    pub fn start(
        &self,
        mut spec: wire::EngineSpec,
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
        let engine = self.next_engine.fetch_add(1, Ordering::Relaxed);
        // A spec of no world starts a world of its own, which takes the
        // identity of its first engine; an engine that replaces a lost one
        // goes on with its world.
        if spec.world == 0 {
            spec.world = engine;
        }
        let world = spec.world;
        let executable = spec
            .workers
            .as_ref()
            .map(|workers| workers.executable.clone())
            .context("the engine spec states no workers")?;
        let standby = spec.predecessor.is_some();
        let mut control = self.spawn(engine, &executable)?;
        let stream = self.accept(engine, &mut control)?;
        let mut reader = Connection::from_unix(stream.try_clone()?)?;
        let mut writer = Connection::from_unix(stream)?;

        let hello: wire::Hello = reader
            .recv(CONNECT_TIMEOUT)?
            .context("the engine process said no hello")?;
        let refusal = if hello.major != proto::PROTOCOL_MAJOR {
            format!(
                "the engine speaks major version {} and this daemon {}",
                hello.major,
                proto::PROTOCOL_MAJOR
            )
        } else if hello.engine != engine {
            format!("engine {} connected for engine {engine}", hello.engine)
        } else {
            String::new()
        };
        writer.send(&wire::Welcome {
            major: proto::PROTOCOL_MAJOR,
            minor: proto::PROTOCOL_MINOR,
            build: proto::BUILD.to_owned(),
            refusal: refusal.clone(),
            spec: refusal.is_empty().then_some(spec),
        })?;
        anyhow::ensure!(refusal.is_empty(), "refused the engine process: {refusal}");
        tracing::info!(engine, build = %hello.build, "the engine process connected; awaiting its start");

        // The engine reports itself starting, then ready or failed. Its
        // process's exit, which the control connection reports, ends the
        // wait too.
        loop {
            match reader.try_recv::<wire::FromEngine>()? {
                Some(wire::FromEngine {
                    kind: Some(wire::from_engine::Kind::State(state)),
                }) => match wire::EngineState::try_from(state.state) {
                    Ok(started @ (wire::EngineState::Ready | wire::EngineState::Standby))
                        if (started == wire::EngineState::Standby) == standby =>
                    {
                        let info = state.info.context("the engine reported no info")?;
                        let state = if standby {
                            tracing::info!(engine, "the engine is prepared beside its predecessor");
                            EngineState::Standby
                        } else {
                            tracing::info!(
                                engine,
                                schedulers = info.schedulers,
                                "the engine serves"
                            );
                            EngineState::Ready
                        };
                        return EngineClient::new(
                            reader,
                            writer,
                            info.into(),
                            state,
                            stats,
                            // The process lives as long as its control
                            // connection.
                            Some(control),
                        )
                        .map(|client| client.numbered(engine).in_world(world));
                    }
                    Ok(wire::EngineState::Failed) => {
                        // The failed engine closes its workers and exits;
                        // their memory is free once it has.
                        await_exit(&mut control, FAILED_EXIT);
                        anyhow::bail!("the engine failed to start: {}", state.error);
                    }
                    _ => {}
                },
                Some(_) => {}
                None => {
                    exited(&mut control)?;
                    reader.wait(POLL_INTERVAL)?;
                }
            }
        }
    }

    /// The request socket's address, which an engine attaches to
    /// (`EngineClient::attach`).
    pub fn request_address(&self) -> &str {
        &self.request_address
    }

    /// The identity the next engine this control starts takes.
    pub fn next_engine(&self) -> u64 {
        self.next_engine.load(Ordering::Relaxed)
    }

    /// Takes over the engine `engine` a daemon this one replaced started,
    /// which serves `served` and lives under `control`: the engine connects
    /// to this daemon's request socket, as the replaced daemon commanded it
    /// (`EngineClient::attach`), is welcomed, and reports itself ready.
    /// Later engines this control starts take the identities from
    /// `next_engine` on. `stats` receives its schedulers' statistics.
    ///
    /// # Errors
    ///
    /// Fails when the engine does not connect in time, speaks another major
    /// version, is another engine, or reports anything but ready.
    pub fn adopt(
        &self,
        engine: u64,
        next_engine: u64,
        served: wire::EngineSpec,
        mut control: Connection,
        stats: StatsSink,
    ) -> anyhow::Result<EngineClient> {
        self.next_engine.fetch_max(next_engine, Ordering::Relaxed);
        let world = served.world;
        let stream = self.accept(engine, &mut control)?;
        let mut reader = Connection::from_unix(stream.try_clone()?)?;
        let mut writer = Connection::from_unix(stream)?;
        let hello: wire::Hello = reader
            .recv(CONNECT_TIMEOUT)?
            .context("the engine said no hello")?;
        let refusal = if hello.major != proto::PROTOCOL_MAJOR {
            format!(
                "the engine speaks major version {} and this daemon {}",
                hello.major,
                proto::PROTOCOL_MAJOR
            )
        } else if hello.engine != engine {
            format!(
                "engine {} attached in place of engine {engine}",
                hello.engine
            )
        } else {
            String::new()
        };
        writer.send(&wire::Welcome {
            major: proto::PROTOCOL_MAJOR,
            minor: proto::PROTOCOL_MINOR,
            build: proto::BUILD.to_owned(),
            refusal: refusal.clone(),
            spec: refusal.is_empty().then_some(served),
        })?;
        anyhow::ensure!(refusal.is_empty(), "refused the engine: {refusal}");
        let deadline = Instant::now() + CONNECT_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            anyhow::ensure!(!remaining.is_zero(), "engine {engine} reported no state");
            let Some(message) = reader.recv::<wire::FromEngine>(remaining)? else {
                continue;
            };
            if let Some(wire::from_engine::Kind::State(state)) = message.kind {
                anyhow::ensure!(
                    state.state == wire::EngineState::Ready as i32,
                    "engine {engine} attached while it does not serve: {}",
                    state.error
                );
                let info = state.info.context("the engine reported no info")?;
                tracing::info!(engine, build = %hello.build, "the engine attached");
                return EngineClient::new(
                    reader,
                    writer,
                    info.into(),
                    EngineState::Ready,
                    stats,
                    Some(control),
                )
                .map(|client| client.numbered(engine).in_world(world));
            }
        }
    }

    /// Connects to the daemon's process socket and has it start engine
    /// `engine` as `executable` on this node; returns the control connection
    /// the process lives under.
    fn spawn(&self, engine: u64, executable: &str) -> anyhow::Result<Connection> {
        let mut control = Connection::connect(&self.processes)
            .with_context(|| format!("connecting to the daemon at {}", self.processes))?;
        control.send(&process::Hello {
            major: proto::PROTOCOL_MAJOR,
            minor: proto::PROTOCOL_MINOR,
            build: proto::BUILD.to_owned(),
            peer: Some(hello::Peer::Engine(process::Engine {
                deployment: self.deployment.clone(),
            })),
        })?;
        let welcome: process::Welcome = control
            .recv(DAEMON_TIMEOUT)?
            .context("the daemon did not answer the control connection")?;
        anyhow::ensure!(
            welcome.refusal.is_empty(),
            "the daemon refused the control connection: {}",
            welcome.refusal
        );

        let executable = executable.to_owned();
        control.send(&process::Upstream {
            kind: Some(upstream::Kind::Request(process::Request {
                id: 1,
                kind: Some(request::Kind::Spawn(process::Spawn {
                    host: self.host.clone(),
                    group: ENGINE_GROUP.to_owned(),
                    rank: 0,
                    generation: engine,
                    executable,
                    arguments: vec![
                        "engine".to_owned(),
                        "--requests".to_owned(),
                        self.request_address.clone(),
                        "--processes".to_owned(),
                        self.processes.clone(),
                        "--engine".to_owned(),
                        engine.to_string(),
                    ],
                    ..process::Spawn::default()
                })),
            })),
        })?;
        let deadline = Instant::now() + DAEMON_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            anyhow::ensure!(
                !remaining.is_zero(),
                "the daemon did not start the engine in time"
            );
            let Some(message) = control.recv::<process::Downstream>(remaining)? else {
                continue;
            };
            if let Some(downstream::Kind::Reply(answer)) = message.kind {
                anyhow::ensure!(
                    answer.error.is_empty(),
                    "starting the engine process: {}",
                    answer.error
                );
                if let Some(reply::Kind::Spawned(spawned)) = answer.kind {
                    tracing::info!(engine, pid = spawned.pid, "started the engine process");
                }
                return Ok(control);
            }
        }
    }

    /// Accepts engine `engine`'s connection to the request socket, failing
    /// when its process exits first or does not connect in time.
    fn accept(&self, engine: u64, control: &mut Connection) -> anyhow::Result<UnixStream> {
        let deadline = Instant::now() + CONNECT_TIMEOUT;
        loop {
            match self.requests.accept() {
                Ok((stream, _)) => return Ok(stream),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(error) => return Err(error).context("accepting the engine's connection"),
            }
            exited(control)?;
            anyhow::ensure!(
                Instant::now() < deadline,
                "engine {engine} did not connect within {CONNECT_TIMEOUT:?}"
            );
            std::thread::sleep(POLL_INTERVAL);
        }
    }
}

impl Drop for EngineControl {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket);
    }
}

/// How long a failed engine takes to close its workers and exit, at most.
const FAILED_EXIT: Duration = Duration::from_secs(120);

/// Waits up to `timeout` for the daemon to report the engine process's exit
/// on its control connection; a wait that times out is logged.
fn await_exit(control: &mut Connection, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        match exited(control) {
            Ok(()) => std::thread::sleep(POLL_INTERVAL),
            Err(_) => return,
        }
    }
    tracing::warn!("the failed engine process did not exit within {timeout:?}");
}

/// Fails when the daemon reported the engine process's exit on its control
/// connection.
fn exited(control: &mut Connection) -> anyhow::Result<()> {
    while let Some(message) = control.try_recv::<process::Downstream>()? {
        if let Some(downstream::Kind::Event(process::Event {
            kind: Some(event::Kind::Exit(exit)),
        })) = message.kind
        {
            anyhow::bail!("the engine process exited: {}", exit.status);
        }
    }
    Ok(())
}
