//! A fake engine at the process interface, and a fake node daemon behind it.
//!
//! The daemon's update drives engines through `proto::process`: commands
//! down, reports up, and requests to the node daemons that start and stop
//! them. A real engine needs devices and a checkpoint, so a test of the
//! update's phase machine drives [`Engine`] instead: it keeps the lifecycle
//! and the generation an engine keeps, acknowledges each command the way the
//! interface states, and records what it was sent. A test states a failure by
//! arming it ([`Engine::refuse_prepare`], [`Engine::fault_at_switch`]) or by
//! sending the report itself ([`Engine::fault`], [`Engine::rolled_back`]),
//! and an engine slower than the others by withholding its acknowledgements
//! ([`Engine::withhold`]).
//!
//! [`node_reply`] replies to the requests a leader makes of a node's daemon
//! (`Spawn`, `Claim`, `Stop`, `KeepWeights`) the way a node does, with what
//! [`Refusals`] names refused.

use proto::process::{self, Command, Report, command, report};
use proto::request::wire::EngineState;

/// A fake engine: its lifecycle, its generations, and what it was sent.
#[derive(Debug, Clone)]
pub struct Engine {
    /// The launch generation it was started at.
    launch_generation: u64,
    lifecycle: EngineState,
    /// The generation of modules it executes.
    generation: u64,
    /// The generation it prepared and has not switched to.
    prepared: Option<u64>,
    /// The generation it keeps while the one it switched to is on trial.
    kept: Option<u64>,
    /// Every command it was sent, in order.
    received: Vec<command::Kind>,
    /// Why its next `Prepare` fails, when armed.
    prepare_refusal: Option<String>,
    /// Why its next `Switch` rolls back by itself, when armed.
    switch_fault: Option<String>,
    /// The acknowledgements of `Promote` and `Drain` it took effect for and
    /// has not sent yet, while it withholds them.
    withheld: Option<Vec<Report>>,
}

impl Engine {
    /// An engine of launch generation `launch_generation` that serves
    /// generation 1, as one started to serve does.
    pub fn serving(launch_generation: u64) -> Self {
        Self::new(launch_generation, EngineState::Serving)
    }

    /// An engine of launch generation `launch_generation` that parked with
    /// generation 1, as one started beside a serving engine does.
    pub fn parked(launch_generation: u64) -> Self {
        Self::new(launch_generation, EngineState::Parked)
    }

    fn new(launch_generation: u64, lifecycle: EngineState) -> Self {
        Self {
            launch_generation,
            lifecycle,
            generation: 1,
            prepared: None,
            kept: None,
            received: Vec::new(),
            prepare_refusal: None,
            switch_fault: None,
            withheld: None,
        }
    }

    /// The launch generation the engine was started at.
    pub fn launch_generation(&self) -> u64 {
        self.launch_generation
    }

    /// Where the engine is in its life.
    pub fn lifecycle(&self) -> EngineState {
        self.lifecycle
    }

    /// The generation of modules the engine executes.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Every command the engine was sent, in order.
    pub fn received(&self) -> &[command::Kind] {
        &self.received
    }

    /// The engine's next `Prepare` fails with `reason`, and nothing changes.
    pub fn refuse_prepare(&mut self, reason: &str) {
        self.prepare_refusal = Some(reason.to_owned());
    }

    /// The engine's next `Switch` fails with `reason`: it goes back to the
    /// kept generation by itself.
    pub fn fault_at_switch(&mut self, reason: &str) {
        self.switch_fault = Some(reason.to_owned());
    }

    /// The engine's acknowledgements of `Promote` and `Drain` wait until
    /// [`Engine::release`]: the commands take effect, and the reports are
    /// late.
    pub fn withhold(&mut self) {
        self.withheld.get_or_insert_with(Vec::new);
    }

    /// The acknowledgements the engine withheld, oldest first; it withholds
    /// nothing more.
    pub fn release(&mut self) -> Vec<Report> {
        self.withheld.take().unwrap_or_default()
    }

    /// What the engine reports once its start completed: that it parked, or
    /// that it serves.
    pub fn started(&self) -> Report {
        match self.lifecycle {
            EngineState::Parked => self.parked_report(),
            _ => self.serving_report(),
        }
    }

    /// A process-fatal fault the engine observed, which it reports before it
    /// exits.
    pub fn fault(&mut self, reason: &str) -> Report {
        self.lifecycle = EngineState::Failed;
        report(report::Kind::Fault(process::Fault {
            kind: "sticky".to_owned(),
            reason: reason.to_owned(),
            device: 0,
        }))
    }

    /// The engine goes back to the kept generation by itself, as it does on a
    /// stalled job or non-finite logits during a trial, and reports it.
    pub fn rolled_back(&mut self, reason: &str) -> Report {
        if let Some(kept) = self.kept.take() {
            self.generation = kept;
        }
        self.prepared = None;
        report(report::Kind::RolledBack(process::RolledBack {
            reason: reason.to_owned(),
        }))
    }

    /// Applies `command` and returns the reports the engine sends for it, in
    /// order.
    pub fn command(&mut self, command: Command) -> Vec<Report> {
        let Some(kind) = command.kind else {
            return Vec::new();
        };
        self.received.push(kind.clone());
        match kind {
            command::Kind::Prepare(_) => match self.prepare_refusal.take() {
                Some(reason) => {
                    vec![report(report::Kind::PrepareFailed(
                        process::PrepareFailed { reason },
                    ))]
                }
                None => {
                    let generation = self.generation + 1;
                    self.prepared = Some(generation);
                    vec![report(report::Kind::Prepared(process::Prepared {
                        generation,
                        uncompared: Vec::new(),
                    }))]
                }
            },
            command::Kind::Switch(switch) => {
                if let Some(reason) = self.switch_fault.take() {
                    return vec![self.rolled_back(&reason)];
                }
                if self.prepared.take() != Some(switch.generation) {
                    return vec![report(report::Kind::Fault(process::Fault {
                        kind: "out_of_order".to_owned(),
                        reason: format!("generation {} was not prepared", switch.generation),
                        device: 0,
                    }))];
                }
                self.kept = Some(std::mem::replace(&mut self.generation, switch.generation));
                vec![report(report::Kind::Switched(process::Switched {
                    generation: switch.generation,
                }))]
            }
            command::Kind::Commit(_) => {
                self.kept = None;
                vec![report(report::Kind::Committed(process::Committed {}))]
            }
            command::Kind::Rollback(rollback) => vec![self.rolled_back(&rollback.reason)],
            command::Kind::Drain(drain) => {
                self.lifecycle = EngineState::Parked;
                self.prepared = None;
                self.acknowledge(report(report::Kind::Drained(process::Drained {
                    epoch: drain.epoch,
                })))
            }
            command::Kind::Promote(_) => {
                self.lifecycle = EngineState::Serving;
                let serving = self.serving_report();
                self.acknowledge(serving)
            }
            command::Kind::Pause(_) | command::Kind::Serve(_) => Vec::new(),
        }
    }

    /// Sends `acknowledgement` now, or keeps it while the engine withholds.
    fn acknowledge(&mut self, acknowledgement: Report) -> Vec<Report> {
        match self.withheld.as_mut() {
            Some(withheld) => {
                withheld.push(acknowledgement);
                Vec::new()
            }
            None => vec![acknowledgement],
        }
    }

    fn parked_report(&self) -> Report {
        report(report::Kind::Parked(process::Parked {
            launch_generation: self.launch_generation,
            generation: self.generation,
        }))
    }

    fn serving_report(&self) -> Report {
        report(report::Kind::Serving(process::Serving {
            launch_generation: self.launch_generation,
            generation: self.generation,
        }))
    }
}

/// One report of `kind`.
fn report(kind: report::Kind) -> Report {
    Report { kind: Some(kind) }
}

/// What a fake node refuses, each with the reason a node's daemon would give.
#[derive(Debug, Clone, Default)]
pub struct Refusals {
    /// Every `Spawn`.
    pub spawn: Option<String>,
    /// Every `Claim`.
    pub claim: Option<String>,
    /// Moving retained weights to host storage: the node keeps them on its
    /// GPUs and says why (`WeightsKept::refused`).
    pub host_storage: Option<String>,
}

/// The bytes of weights a fake node moves to host storage when it is asked to
/// and does not refuse.
pub const MOVED_BYTES: u64 = 1 << 30;

/// The process id a fake node gives the engine of launch generation
/// `generation`.
pub fn pid(generation: u64) -> u32 {
    10_000 + u32::try_from(generation).unwrap_or(0)
}

/// The reply of a node's daemon to `request`, refusing what `refusals` names.
pub fn node_reply(request: &process::Request, refusals: &Refusals) -> process::Reply {
    use process::{reply, request::Kind};

    let mut reply = process::Reply {
        id: request.id,
        ..process::Reply::default()
    };
    match (&request.kind, refusals) {
        (
            Some(Kind::Spawn(_)),
            Refusals {
                spawn: Some(why), ..
            },
        )
        | (
            Some(Kind::Claim(_)),
            Refusals {
                claim: Some(why), ..
            },
        ) => reply.error.clone_from(why),
        (Some(Kind::Spawn(spawn)), _) => {
            reply.kind = Some(reply::Kind::Spawned(process::Spawned {
                pid: pid(spawn.generation),
            }));
        }
        (Some(Kind::Claim(_)), _) => {
            reply.kind = Some(reply::Kind::Claimed(process::Claimed {
                socket: "/run/storage.sock".to_owned(),
                claim: Vec::new(),
                tcp: String::new(),
            }));
        }
        (Some(Kind::Stop(_)), _) => {
            reply.kind = Some(reply::Kind::Stopped(process::Stopped {}));
        }
        (Some(Kind::KeepWeights(keep)), _) => {
            let refused = keep
                .to_host
                .then(|| refusals.host_storage.clone())
                .flatten();
            let moved = keep.to_host && refused.is_none();
            reply.kind = Some(reply::Kind::WeightsKept(process::WeightsKept {
                released: 1,
                moved_bytes: if moved { MOVED_BYTES } else { 0 },
                refused: refused.unwrap_or_default(),
            }));
        }
        (other, _) => reply.error = format!("a node replies to no {other:?}"),
    }
    reply
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(kind: command::Kind) -> Command {
        Command { kind: Some(kind) }
    }

    /// A serving engine prepares, switches and commits a generation, each
    /// acknowledged as the interface states, and keeps the one it left until
    /// the commit.
    #[test]
    fn a_generation_is_prepared_switched_and_committed() {
        let mut engine = Engine::serving(7);
        let prepared = engine.command(kind(command::Kind::Prepare(process::Prepare::default())));
        assert_eq!(
            prepared[0].kind,
            Some(report::Kind::Prepared(process::Prepared {
                generation: 2,
                uncompared: Vec::new(),
            }))
        );
        assert_eq!(engine.generation(), 1, "serving is untouched");

        let switched = engine.command(kind(command::Kind::Switch(process::Switch {
            generation: 2,
        })));
        assert_eq!(
            switched[0].kind,
            Some(report::Kind::Switched(process::Switched { generation: 2 }))
        );
        assert_eq!(engine.generation(), 2);

        engine.command(kind(command::Kind::Commit(process::Commit {})));
        let rolled = engine.command(kind(command::Kind::Rollback(process::Rollback::default())));
        assert!(matches!(rolled[0].kind, Some(report::Kind::RolledBack(_))));
        assert_eq!(engine.generation(), 2, "nothing is kept after a commit");
    }

    /// A rollback returns to the kept generation, and one that finds nothing
    /// kept is acknowledged all the same.
    #[test]
    fn a_rollback_returns_to_the_kept_generation() {
        let mut engine = Engine::serving(7);
        engine.command(kind(command::Kind::Prepare(process::Prepare::default())));
        engine.command(kind(command::Kind::Switch(process::Switch {
            generation: 2,
        })));
        let rolled = engine.command(kind(command::Kind::Rollback(process::Rollback {
            reason: "a threshold".to_owned(),
        })));
        assert_eq!(
            rolled[0].kind,
            Some(report::Kind::RolledBack(process::RolledBack {
                reason: "a threshold".to_owned(),
            }))
        );
        assert_eq!(engine.generation(), 1);
    }

    /// A drain parks the engine and a promotion makes it serve, each
    /// acknowledged with the launch generation the engine was started at.
    #[test]
    fn a_drain_parks_and_a_promotion_serves() {
        let mut engine = Engine::serving(7);
        let drained = engine.command(kind(command::Kind::Drain(process::Drain {
            epoch: 3,
            ..Default::default()
        })));
        assert_eq!(
            drained[0].kind,
            Some(report::Kind::Drained(process::Drained { epoch: 3 }))
        );
        assert_eq!(engine.lifecycle(), EngineState::Parked);

        let serving = engine.command(kind(command::Kind::Promote(process::Promote::default())));
        assert_eq!(
            serving[0].kind,
            Some(report::Kind::Serving(process::Serving {
                launch_generation: 7,
                generation: 1,
            }))
        );
        assert_eq!(engine.lifecycle(), EngineState::Serving);
    }
}
