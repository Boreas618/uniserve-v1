//! What a main process starts from: the launch file, the handshake on the daemon's request
//! interface that states the epoch ([`Epoch`]), and the connection to the daemon's process
//! interface ([`ProcessInterface`]) that carries the supervision frames.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use proto::config::EngineArgs;
use worker::control::Control;

/// The epoch as this node takes it at start.
///
/// The daemon fixes the epoch with a [`proto::config::EngineConfig`] and
/// fills its operators and its scheduling interface with programs; the
/// main process never receives program content, only the programs' names, and it
/// resolves their modules from the Program Server itself.
pub(crate) struct Epoch {
    /// The program revision the daemon numbered for the generation that serves
    /// once this node switches over. Zero until the daemon's first `Precopy` names the
    /// number it prepares: the number is the daemon's and this process never invents one.
    pub generation: u64,
    /// The host identity of the node this process runs on, which the daemon knows and which the
    /// Storage Server's claim states.
    pub host: String,
    /// The epoch this process succeeds, whose storages it maps until it is promoted.
    pub predecessor_epoch: Option<u64>,
    /// The routed experts' distribution this epoch loads: what a redistribution committed,
    /// which a relaunch and a standby load again.
    pub experts: BTreeMap<String, Vec<u32>>,
    /// The deployment's always-on profiling (`profiling_config`).
    pub profiling: proto::config::ProfilingConfig,
    /// The KV cache the epoch this process succeeds holds, by worker group.
    pub predecessor_cache: Vec<proto::request::wire::CacheSegments>,
}

/// The launch file's contents.
pub(super) fn read_engine_args(path: &Path) -> Result<EngineArgs> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening the launch file {}", path.display()))?;
    serde_json::from_reader(file)
        .with_context(|| format!("reading the launch file {}", path.display()))
}

/// This process's connection to its daemon's process interface.
///
/// The daemon has two interfaces and they are two sockets with two vocabularies: the request one
/// carries the epoch's requests, their events and the metrics, and this one carries the supervision
/// frames - the storage claim, the program-layer reports, and every command the daemon sends an
/// epoch. A frame of one sent on the other is unreadable at the far end, which is why the claim and
/// the reports live here and not on the request connection.
pub(super) struct ProcessInterface {
    pub(super) connection: proto::Connection,
    /// The commands read while waiting for the answer to something else. A daemon may send a
    /// `Precopy` while a claim is outstanding, and dropping it would lose the epoch's first
    /// generation.
    pending: std::collections::VecDeque<proto::process::Command>,
    /// The daemon closed the interface: the epoch is over, and the process says so and goes.
    closed: bool,
}

impl ProcessInterface {
    /// Connects to `address` and states this process's identity, which the daemon answers with the
    /// deployment's nodes or a refusal.
    pub(super) fn connect(address: &str, deployment: &str, engine: u64) -> Result<Self> {
        // A process link that cannot name its engine is refused here rather than sent: zero states
        // "no main process", and a daemon that accepted it would file this connection in the place
        // of a process it can never command. The launch file states the number, so a launch without
        // one is the deployment's fault and this is where it is named.
        if engine == 0 {
            bail!(
                "the launch file states engine 0, which names no main process: the daemon addresses \
                 the epoch's program-layer commands by this number, so a process it started under \
                 none cannot be commanded"
            );
        }
        let mut connection = proto::Connection::connect(address).with_context(|| {
            format!("connecting to the daemon's process interface at {address}")
        })?;
        connection.send(&proto::process::Hello {
            major: proto::PROTOCOL_MAJOR,
            build: proto::BUILD.to_owned(),
            peer: Some(proto::process::hello::Peer::Engine(
                proto::process::Engine {
                    deployment: deployment.to_owned(),
                    // The daemon keys this connection by the number the epoch layer started this
                    // process under, so a command it sends for engine N reaches this process and
                    // not whichever connection the daemon happened to accept Nth.
                    engine,
                    // This is the process link: the one the daemon's program-layer commands
                    // travel down. The request link, on its own socket, states `REQUESTS`, so the
                    // two links of one engine are never confused for one another.
                    interface: proto::process::Interface::Process as i32,
                },
            )),
        })?;
        // The welcome is the daemon's acceptance of this process: a node that has not admitted it
        // would mint a claim for a process it does not supervise, so nothing is asked before it
        // arrives.
        let welcome: proto::process::Welcome = connection
            .recv(HANDSHAKE_TIMEOUT)?
            .context("the daemon did not answer this main process's process handshake")?;
        if !welcome.refusal.is_empty() {
            bail!(
                "the daemon refused this main process's process connection: {}",
                welcome.refusal
            );
        }
        // A process the daemon admits and lists no node for is a refusal too: the nodes are what the
        // claim and every later request are answered by, so a welcome with none would let this
        // process run on a deployment the daemon does not know it on.
        if welcome.nodes.is_empty() {
            bail!(
                "the daemon admitted this main process but listed no node of the deployment, so \
                 nothing here can be claimed or supervised"
            );
        }
        tracing::info!(
            nodes = welcome.nodes.len(),
            build = %welcome.build,
            // The engine number this link claims, so this process's line and the daemon's
            // "a process connected engine=N" are matched by a number rather than by their order in
            // the log - which is the confusion that cost a run.
            engine,
            interface = "process",
            "connected to the deployment's process interface"
        );
        Ok(Self {
            connection,
            pending: std::collections::VecDeque::new(),
            closed: false,
        })
    }

    /// Receives one `Downstream` message, waiting no longer than `timeout`.
    fn receive(
        connection: &mut proto::Connection,
        timeout: Duration,
        what: &str,
    ) -> Result<proto::process::Downstream> {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                bail!("the daemon did not answer {what} within {timeout:?}");
            }
            connection.wait(remaining)?;
            if let Some(message) = connection.try_recv::<proto::process::Downstream>()? {
                return Ok(message);
            }
        }
    }

    /// Sends one request and answers its reply, filing every command that arrives meanwhile.
    pub(super) fn request(
        &mut self,
        request: proto::process::Request,
        timeout: Duration,
    ) -> Result<proto::process::Reply> {
        let id = request.id;
        self.connection.send(&proto::process::Upstream {
            kind: Some(proto::process::upstream::Kind::Request(request)),
        })?;
        loop {
            let message = Self::receive(&mut self.connection, timeout, "a claim")?;
            match message.kind {
                Some(proto::process::downstream::Kind::Reply(reply)) if reply.id == id => {
                    return Ok(reply);
                }
                Some(proto::process::downstream::Kind::Reply(reply)) => {
                    tracing::warn!(
                        id = reply.id,
                        "an answer to a request this process did not send"
                    );
                }
                Some(proto::process::downstream::Kind::Event(event)) => self.file(event),
                None => {}
            }
        }
    }

    /// Sends one report of this process's state, which the daemon needs no answer to.
    pub(super) fn report(&mut self, report: proto::process::Report) -> Result<()> {
        self.connection.send(&proto::process::Upstream {
            kind: Some(proto::process::upstream::Kind::Report(report)),
        })?;
        Ok(())
    }

    /// The daemon's next command for this epoch, if one has arrived.
    ///
    /// Nothing waits here: a command that has not arrived is one the next serve pass looks for.
    pub(super) fn take_command(&mut self) -> Result<Option<proto::process::Command>> {
        // The daemon owns this process's life: the interface closing is the end of the epoch and not
        // an error, so it stops the loop by name instead of surfacing as a failure the daemon never
        // sees. A process that ends silently on a closed interface is how a run reads as a lost
        // engine with no side of its own in the log.
        loop {
            let message = match self.connection.try_recv::<proto::process::Downstream>() {
                Ok(Some(message)) => message,
                Ok(None) => break,
                Err(proto::Error::Closed) => {
                    tracing::warn!(
                        "the daemon closed this process's interface; whether this epoch ends \
                         depends on whether another daemon holds it"
                    );
                    self.closed = true;
                    return Ok(None);
                }
                Err(error) => return Err(error.into()),
            };
            match message.kind {
                Some(proto::process::downstream::Kind::Event(event)) => self.file(event),
                Some(proto::process::downstream::Kind::Reply(reply)) => {
                    tracing::warn!(
                        id = reply.id,
                        "an answer to a request this process did not send"
                    );
                }
                None => {}
            }
        }
        Ok(self.pending.pop_front())
    }

    /// Whether the daemon closed the interface, which ends this process.
    pub(super) const fn closed(&self) -> bool {
        self.closed
    }

    /// Files what the daemon sent unasked.
    fn file(&mut self, event: proto::process::Event) {
        match event.kind {
            Some(proto::process::event::Kind::Command(command)) => self.pending.push_back(command),
            // A group's exit is a worker's report; this process supervises no group.
            Some(proto::process::event::Kind::Exit(exit)) => {
                tracing::warn!(
                    group = %exit.group,
                    rank = exit.rank,
                    status = %exit.status,
                    "an exit report names a group this process does not supervise"
                );
            }
            None => {}
        }
    }
}

/// How long the daemon's process interface may take to answer the handshake, which it answers at
/// once: the wait exists to fail a wrong address rather than hang the process.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

/// The handshake with the daemon, which answers with the epoch this node serves.
///
/// The daemon states that this build may serve the epoch, and, later, the program revision of the
/// generation it prepares. The launch file states the rest of what the epoch starts from:
/// this node's host identity, the process this one succeeds, the expert order and the profiling.
/// The programs themselves never come through this interface: the daemon names them and this
/// process resolves their modules from the Program Server.
pub(super) fn handshake(control: &mut Control, launch: &EngineArgs) -> Result<Epoch> {
    control.daemon_mut().send(&proto::request::wire::Hello {
        engine: launch.engine,
        build: proto::BUILD.to_owned(),
        major: proto::PROTOCOL_MAJOR,
        // The request link: requests, their events and the epoch's metrics. The process link
        // states `PROCESS` on its own socket, and both name the same engine number.
        interface: proto::process::Interface::Requests as i32,
    })?;
    let welcome: proto::request::wire::Welcome = control
        .daemon_mut()
        .recv(Duration::from_secs(60))?
        .context("the daemon closed the request interface during the handshake")?;
    if !welcome.refusal.is_empty() {
        bail!("the daemon refused this main process: {}", welcome.refusal);
    }

    // The epoch's programs and the generation that serves at start are not in the launch file:
    // the programs arrive as the daemon's first `Precopy` and the generation number
    // is the daemon's own. Until it states one, this process serves generation zero, whose
    // `Precopy` names the daemon's next number.
    //
    // The host identity is the daemon's fact and nothing else can supply it: the Storage Server
    // keys this rank's claim by it, so an empty one would mint a claim on a host that does not
    // exist and every peer would compare its products against the wrong node.
    if launch.host.is_empty() {
        bail!(
            "the launch file states no host identity, which this rank's storage claim is keyed by"
        );
    }
    let predecessor = launch.predecessor.as_ref();
    Ok(Epoch {
        generation: 0,
        predecessor_epoch: predecessor.map(|predecessor| predecessor.epoch),
        predecessor_cache: predecessor
            .map(|predecessor| predecessor.segments.clone())
            .unwrap_or_default(),
        experts: launch
            .expert_order
            .iter()
            .map(|(module, order)| (module.clone(), order.experts.clone()))
            .collect(),
        host: launch.host.clone(),
        profiling: launch.engine_config.profiling.clone(),
    })
}
