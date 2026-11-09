//! Stand-in main processes behind the daemon's command channel.
//!
//! The program layer's tests need a daemon that commands something: an
//! engine's own process needs a device and a checkpoint, so what answers here
//! is the protocol alone — each command is answered the way §5.1 says a main
//! process answers it, and what the daemon sent is recorded for the test to
//! read. Test code only (`#[cfg(test)]`).

use std::sync::{Arc, Mutex};

use proto::process::{self, command, report};

use crate::engine::control::EpochProcess;
use crate::supervision::ProcessCommands;

/// What a stand-in main process saw a daemon send it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Seen {
    Precopy,
    Switchover(u64),
    Commit,
    Revert(String),
    Purge(u64),
}

/// How a stand-in main process answers the commands it is sent.
pub(crate) enum Answer {
    /// It serves, and numbers this generation for the precopy it answers.
    Serves(u64),
    /// It cannot prepare the programs, with this reason.
    Refuses(String),
    /// It reports this instead of answering the precopy.
    Reports(fn() -> process::Report),
}

/// What a stand-in main process was handed by a `Precopy`: the programs it
/// would have loaded, as the epoch names them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrecopyPayload {
    /// The operators the command carried.
    pub(crate) kernels: usize,
    /// The revision each operator's program stated, by operator: a joining
    /// process must receive the numbers the epoch installed, never the zero a
    /// filled program carries before a Program Server selects one.
    pub(crate) revisions: Vec<(String, u32)>,
    /// The scheduling program's backend.
    pub(crate) policy: String,
}

/// The service's end of the daemon's channel, with stand-in main processes
/// behind it: every command is answered as the protocol says, recording what
/// each engine was sent.
#[expect(
    clippy::type_complexity,
    reason = "a stand-in hands back its commands and the record of what it saw, once"
)]
pub(crate) fn main_processes(
    answers: Vec<(u64, Answer)>,
) -> (ProcessCommands, Arc<Mutex<Vec<(u64, Seen)>>>) {
    let (commands, seen, _) = main_processes_recording(answers);
    (commands, seen)
}

/// As [`main_processes`], and the programs each `Precopy` carried: a joining
/// main process must be handed the epoch's own programs, not none.
#[expect(
    clippy::type_complexity,
    reason = "a stand-in hands back its commands and the record of what it saw, once"
)]
pub(crate) fn main_processes_recording(
    answers: Vec<(u64, Answer)>,
) -> (
    ProcessCommands,
    Arc<Mutex<Vec<(u64, Seen)>>>,
    Arc<Mutex<Vec<(u64, PrecopyPayload)>>>,
) {
    let answers: std::collections::BTreeMap<u64, Answer> = answers.into_iter().collect();
    let (processes, commands, reports) = ProcessCommands::channel();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&seen);
    let payloads = Arc::new(Mutex::new(Vec::new()));
    let carried = Arc::clone(&payloads);
    std::thread::spawn(move || {
        while let Ok((engine, command)) = commands.recv() {
            let answer = answers.get(&engine);
            let generation = match answer {
                Some(Answer::Serves(generation)) => *generation,
                _ => 0,
            };
            let (what, reply) = match command.kind {
                Some(command::Kind::Precopy(precopy)) => {
                    // What the command carried, recorded before the answer:
                    // the test asks what a process was handed, whether or not
                    // it took it.
                    carried.lock().unwrap().push((
                        engine,
                        PrecopyPayload {
                            kernels: precopy.kernels.len(),
                            revisions: precopy
                                .kernels
                                .iter()
                                .map(|kernel| {
                                    (
                                        kernel.op.clone(),
                                        kernel.program.as_ref().map_or(0, |p| p.revision),
                                    )
                                })
                                .collect(),
                            policy: precopy
                                .policy
                                .as_ref()
                                .map(|policy| policy.backend.clone())
                                .unwrap_or_default(),
                        },
                    ));
                    match answer {
                        Some(Answer::Serves(_)) => (
                            Seen::Precopy,
                            Some(process::Report {
                                kind: Some(report::Kind::Precopied(process::Precopied {
                                    generation,
                                    uncompared: Vec::new(),
                                })),
                            }),
                        ),
                        Some(Answer::Refuses(reason)) => (
                            Seen::Precopy,
                            Some(process::Report {
                                kind: Some(report::Kind::PrecopyFailed(process::PrecopyFailed {
                                    reason: reason.clone(),
                                })),
                            }),
                        ),
                        Some(Answer::Reports(report)) => (Seen::Precopy, Some(report())),
                        None => (Seen::Precopy, None),
                    }
                }
                Some(command::Kind::Switchover(switchover)) => (
                    Seen::Switchover(switchover.generation),
                    Some(process::Report {
                        kind: Some(report::Kind::Switched(process::Switched {
                            generation: switchover.generation,
                        })),
                    }),
                ),
                Some(command::Kind::Commit(_)) => (
                    Seen::Commit,
                    Some(process::Report {
                        kind: Some(report::Kind::Committed(process::Committed {})),
                    }),
                ),
                Some(command::Kind::Revert(revert)) => (
                    Seen::Revert(revert.reason.clone()),
                    Some(process::Report {
                        kind: Some(report::Kind::Reverted(process::Reverted {
                            reason: String::new(),
                        })),
                    }),
                ),
                Some(command::Kind::Purge(purge)) => (
                    Seen::Purge(purge.generation),
                    Some(process::Report {
                        kind: Some(report::Kind::Purged(process::Purged {
                            generation: purge.generation,
                        })),
                    }),
                ),
                other => panic!("the switch sent a command it does not send: {other:?}"),
            };
            recorded.lock().unwrap().push((engine, what));
            if let Some(reply) = reply {
                let _ = reports.send((engine, reply));
            }
        }
    });
    (processes, seen, payloads)
}

/// A main process of `host` registered as engine `engine`.
pub(crate) fn process(host: &str, engine: u64) -> EpochProcess {
    EpochProcess {
        host: host.to_owned(),
        engine,
    }
}
