//! A stand-in for the deployment's node daemons and the main processes behind
//! the epoch layer's link (`roster::EpochLink`).
//!
//! The epoch layer's tests drive the real [`ProcessRoster`] at its process
//! boundary: what it asks of the nodes (`Spawn`, `Claim`, `Stop`,
//! `KeepWeights`) is answered here the way a node's daemon answers it and kept
//! for the test to read, and the test plays each main process — its greeting,
//! its reports and its exit — and reads the commands the daemon sent it. No
//! process starts and no storage is claimed. Test code only (`#[cfg(test)]`).

use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use proto::process::{self, Exit, Report, Request, request};

use crate::epoch::Epochs;
use crate::supervision::roster::{EpochLink, Greeting, ProcessRoster};
use crate::supervision::{CommandsDown, ProcessCommands};

/// What the stand-in nodes refuse, each with the reason a node's daemon would
/// give.
#[derive(Debug, Clone, Default)]
pub(crate) struct Refusals {
    /// Every `Spawn`.
    pub(crate) spawn: Option<String>,
    /// Every `Claim`.
    pub(crate) claim: Option<String>,
    /// Moving retained weights to host storage: the node keeps them on its
    /// GPUs and says why (`WeightsKept::refused`).
    pub(crate) host_storage: Option<String>,
}

/// The bytes of weights a node moves to host storage when it is asked to and
/// does not refuse.
pub(crate) const MOVED_BYTES: u64 = 1 << 30;

/// The nodes' and the main processes' ends of one epoch link.
pub(crate) struct Nodes {
    /// Every request the epoch layer made of the nodes, in arrival order.
    requests: Arc<Mutex<Vec<request::Kind>>>,
    /// The commands the daemon sent its peers, by connection.
    commands: CommandsDown,
    /// The reports a peer sends up, by connection.
    reports: std::sync::mpsc::Sender<(u64, Report)>,
    exits: Sender<Exit>,
    greetings: Sender<(u64, Greeting)>,
}

/// The stand-in nodes behind a new epoch link, refusing what `refusals`
/// names.
///
/// A request is recorded before it is answered, and the epoch layer waits for
/// every answer, so a request is readable as soon as the call that made it
/// returned.
pub(crate) fn link(refusals: Refusals) -> (Nodes, EpochLink) {
    let (requests, incoming) = std::sync::mpsc::channel::<(Request, Sender<process::Reply>)>();
    let (exits, exited) = std::sync::mpsc::channel();
    let (greetings, greeted) = std::sync::mpsc::channel();
    let log = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&log);
    std::thread::spawn(move || {
        while let Ok((request, answer)) = incoming.recv() {
            let Some(kind) = request.kind else {
                continue;
            };
            recorded.lock().unwrap().push(kind.clone());
            let mut reply = process::Reply {
                id: request.id,
                ..process::Reply::default()
            };
            match (&kind, &refusals) {
                (
                    request::Kind::Spawn(_),
                    Refusals {
                        spawn: Some(why), ..
                    },
                )
                | (
                    request::Kind::Claim(_),
                    Refusals {
                        claim: Some(why), ..
                    },
                ) => {
                    reply.error = why.clone();
                }
                (request::Kind::Spawn(spawn), _) => {
                    reply.kind = Some(process::reply::Kind::Spawned(process::Spawned {
                        pid: pid(spawn.generation),
                    }));
                }
                (request::Kind::Claim(_), _) => {
                    reply.kind = Some(process::reply::Kind::Claimed(process::Claimed {
                        socket: "/run/storage.sock".to_owned(),
                        claim: Vec::new(),
                        tcp: String::new(),
                    }));
                }
                (request::Kind::Stop(_), _) => {
                    reply.kind = Some(process::reply::Kind::Stopped(process::Stopped {}));
                }
                (request::Kind::KeepWeights(keep), _) => {
                    let refused = keep
                        .to_host
                        .then(|| refusals.host_storage.clone())
                        .flatten();
                    let moved = keep.to_host && refused.is_none();
                    reply.kind = Some(process::reply::Kind::WeightsKept(process::WeightsKept {
                        released: 1,
                        moved_bytes: if moved { MOVED_BYTES } else { 0 },
                        refused: refused.unwrap_or_default(),
                    }));
                }
                (other, _) => reply.error = format!("a node answers no {other:?}"),
            }
            // The epoch layer may have given up on an answer; the record of
            // its request stands either way.
            let _ = answer.send(reply);
        }
    });
    let (processes, commands, reports) = ProcessCommands::pair();
    let link = EpochLink {
        requests,
        greetings: greeted,
        exits: exited,
        commands: processes,
    };
    let nodes = Nodes {
        requests: log,
        commands,
        reports,
        exits,
        greetings,
    };
    (nodes, link)
}

/// The real roster of `deployment`'s epoch layer over stand-in nodes.
pub(crate) fn roster(epochs: &Arc<Epochs>, refusals: Refusals) -> (Nodes, ProcessRoster) {
    let (nodes, link) = link(refusals);
    let roster = ProcessRoster::new(
        Arc::clone(epochs),
        "deployment".to_owned(),
        Vec::new(),
        link,
    );
    (nodes, roster)
}

/// The process id a node gives the child of launch generation `generation`.
pub(crate) fn pid(generation: u64) -> u32 {
    4000 + u32::try_from(generation).expect("a small generation")
}

impl Nodes {
    /// Every request the epoch layer made of the nodes so far.
    pub(crate) fn requests(&self) -> Vec<request::Kind> {
        self.requests.lock().unwrap().clone()
    }

    /// The processes the nodes were asked to start.
    pub(crate) fn spawns(&self) -> Vec<process::Spawn> {
        self.requests()
            .into_iter()
            .filter_map(|kind| match kind {
                request::Kind::Spawn(spawn) => Some(spawn),
                _ => None,
            })
            .collect()
    }

    /// The claims the nodes were asked to hold.
    pub(crate) fn claims(&self) -> Vec<process::Claim> {
        self.requests()
            .into_iter()
            .filter_map(|kind| match kind {
                request::Kind::Claim(claim) => Some(claim),
                _ => None,
            })
            .collect()
    }

    /// The launch generations the nodes were asked to stop; `None` is every
    /// generation of a group.
    pub(crate) fn stops(&self) -> Vec<Option<u64>> {
        self.requests()
            .into_iter()
            .filter_map(|kind| match kind {
                request::Kind::Stop(stop) => Some(stop.generation),
                _ => None,
            })
            .collect()
    }

    /// The retentions of a previous epoch's weights the nodes were asked for.
    pub(crate) fn retentions(&self) -> Vec<process::KeepWeights> {
        self.requests()
            .into_iter()
            .filter_map(|kind| match kind {
                request::Kind::KeepWeights(keep) => Some(keep),
                _ => None,
            })
            .collect()
    }

    /// The commands the daemon sent since the last call, by connection, in
    /// the order it sent them.
    pub(crate) fn sent(&self) -> Vec<(u64, process::command::Kind)> {
        std::iter::from_fn(|| self.commands.try_recv().ok())
            .map(|(peer, command)| (peer, command.kind.expect("a command states its kind")))
            .collect()
    }

    /// The peer on connection `peer` greets the leader as a main process of
    /// `build`.
    pub(crate) fn greet(&self, peer: u64, build: &str) {
        self.greetings
            .send((
                peer,
                Greeting {
                    build: build.to_owned(),
                    frontend: false,
                },
            ))
            .expect("the epoch layer reads greetings");
    }

    /// The frontend greets the leader on connection `peer`.
    pub(crate) fn greet_frontend(&self, peer: u64) {
        self.greetings
            .send((
                peer,
                Greeting {
                    build: proto::BUILD.to_owned(),
                    frontend: true,
                },
            ))
            .expect("the epoch layer reads greetings");
    }

    /// The peer on connection `peer` reports `kind`.
    pub(crate) fn report(&self, peer: u64, kind: process::report::Kind) {
        self.reports
            .send((peer, Report { kind: Some(kind) }))
            .expect("the epoch layer reads reports");
    }

    /// The node `host` reports that the process of launch generation
    /// `generation` exited with `status`.
    pub(crate) fn exit(&self, host: &str, generation: u64, status: &str) {
        self.exits
            .send(Exit {
                host: host.to_owned(),
                group: "engine".to_owned(),
                rank: 0,
                generation,
                status: status.to_owned(),
                last_line: String::new(),
            })
            .expect("the epoch layer reads exits");
    }
}
