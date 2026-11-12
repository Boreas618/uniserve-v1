//! Where an engine is: its lifecycle, and while it serves, its generation update.
//!
//! [`Lifecycle`] states where the engine is in its life and [`Generation`] where a serving
//! engine's generation update is. [`State`] holds both and is moved only by messages: a command of
//! the daemon's process interface ([`State::command`]), the completion of the work a command
//! started (`started`, `prepared`, `switched`, `drained`, `promoted`, ...), or a failure the
//! engine observed ([`State::failure`]). Each command returns the [`Step`] the control thread
//! takes for it, and each completion the [`Report`] that acknowledges it, so the rules are stated
//! here once and nothing in them touches a device, a scheduler or a socket.
//!
//! | From | Message | To | Report |
//! | --- | --- | --- | --- |
//! | `Starting` | start completed, launched parked | `Parked` | `Parked` |
//! | `Starting` | start completed | `Serving` | `Serving` |
//! | `Parked` | `Promote` | `Serving` | `Serving` |
//! | `Serving` | `Drain` | `Draining` | — |
//! | `Draining` | every scheduler parked | `Parked` | `Drained` |
//! | `Draining` | `Promote` | `Draining`, then `Serving` once parked | `Drained`, `Serving` |
//! | any | shutdown | `Stopped` | — |
//! | any | a process-fatal failure | `Failed` | the failure's own |
//!
//! | From | Message | To | Report |
//! | --- | --- | --- | --- |
//! | `Steady` | `Prepare` | `Preparing` | — |
//! | `Preparing` | preparation finished | `Prepared` | `Prepared` |
//! | `Preparing` | preparation failed | `Steady` | `PrepareFailed` |
//! | `Prepared` | `Switch` | `Trial` | `Switched`, once its captures finished |
//! | `Trial` | `Commit` | `Steady` | `Committed` |
//! | `Trial` | `Rollback`, or a failure that is not process-fatal | `Steady`, on the kept generation | `RolledBack` |
//! | `Preparing`, `Prepared` | `Rollback`, `Drain` | `Steady` | `RolledBack` |
//! | `Steady` | `Commit`, `Rollback` | `Steady` | `Committed`, `RolledBack` |

use ipc::UpdateFailure;
use proto::process::{Report, command, report};
use proto::request::wire::EngineState;

use super::generation::{Action, Generation, Generations};

/// Where an engine is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lifecycle {
    /// It loads its ranks, resolves the modules its launch file names and captures its graphs.
    Starting,
    /// It executes nothing and admits nothing, with its storage mapped and its graphs captured:
    /// started beside the serving engine (a switch's target, a standby), or drained.
    Parked,
    /// It admits and executes requests.
    Serving,
    /// It admits nothing more and its requests in flight end as migratable at their step
    /// boundaries; parked once every scheduler parked.
    Draining {
        /// A `Promote` arrived meanwhile: the engine serves again as soon as it is parked.
        serves_again: bool,
    },
    /// It was shut down.
    Stopped,
    /// It failed and serves nothing more.
    Failed,
}

impl Lifecycle {
    /// The state as the request link announces it.
    pub(crate) fn state(self) -> EngineState {
        match self {
            Self::Starting => EngineState::Starting,
            Self::Parked => EngineState::Parked,
            Self::Serving => EngineState::Serving,
            Self::Draining { .. } => EngineState::Draining,
            Self::Stopped => EngineState::Stopped,
            Self::Failed => EngineState::Failed,
        }
    }

    /// Why a submission is refused in this state; `None` when the engine admits.
    pub(crate) fn refusal(self) -> Option<&'static str> {
        match self {
            Self::Serving => None,
            Self::Starting => Some("the engine is starting and serves nothing yet"),
            Self::Parked => Some("the engine is parked and serves nothing"),
            Self::Draining { .. } => Some("the engine is draining and admits nothing"),
            Self::Stopped | Self::Failed => Some("the engine serves nothing more"),
        }
    }
}

/// What the control thread does for a command the state accepted.
///
/// A step whose work completes later is acknowledged by the completion's own method of
/// [`State`]; `Report` is a command that needs no work, with the acknowledgement or the refusal
/// it is acknowledged or refused with.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Step {
    /// Prepare the packages the command names as `generation`.
    Prepare { generation: u64 },
    /// Switch the devices and the schedulers to `generation` at a batch boundary, then
    /// [`State::switch`].
    Switch { generation: u64 },
    /// Commit the policy and release `kept`, the generation the trial kept.
    Commit { kept: u64 },
    /// Leave generation `left`: switch back to the kept one when it `executed`, discard what was
    /// prepared otherwise.
    Rollback { left: u64, executed: bool },
    /// Park every scheduler; `discarded` is a generation that was being prepared or was prepared,
    /// which no switch can take any more.
    Drain { epoch: u64, discarded: Option<u64> },
    /// Take the handoff the command states, if any, and admit.
    Promote,
    /// Hold or release admissions.
    Pause(bool),
    /// Send this report; nothing else changes.
    Report(Report),
    /// The command is not this engine's.
    Ignore,
}

/// An engine's lifecycle and generation update, moved by messages.
#[derive(Debug)]
pub(crate) struct State {
    lifecycle: Lifecycle,
    pub(super) generations: Generations,
    /// The launch generation the daemon started this engine at, which every lifecycle report
    /// states: the daemon binds this engine's connection to its launch by it.
    launch_generation: u64,
}

impl State {
    /// A starting engine, launched at `launch_generation`.
    pub(crate) fn new(launch_generation: u64, generations: Generations) -> Self {
        Self {
            lifecycle: Lifecycle::Starting,
            generations,
            launch_generation,
        }
    }

    /// Where the engine is in its life.
    pub(crate) fn lifecycle(&self) -> Lifecycle {
        self.lifecycle
    }

    /// Where its generation update is.
    pub(crate) fn update(&self) -> Generation {
        self.generations.update()
    }

    /// Applies one command of the daemon's process interface and returns what the control
    /// thread does for it.
    pub(crate) fn command(&mut self, kind: &command::Kind) -> Step {
        match kind {
            command::Kind::Prepare(_) => self.prepare(),
            command::Kind::Switch(switch) => self.accept_switch(switch.generation),
            command::Kind::Commit(_) => self.commit(),
            command::Kind::Rollback(rollback) => self.rollback(&rollback.reason),
            command::Kind::Drain(drain) => self.drain(drain.epoch),
            command::Kind::Promote(_) => self.promote(),
            command::Kind::Pause(pause) => Step::Pause(pause.paused),
            // A frontend's command: an engine is told where requests go by nobody.
            command::Kind::Serve(_) => Step::Ignore,
        }
    }

    fn prepare(&mut self) -> Step {
        if self.lifecycle != Lifecycle::Serving {
            return Step::Report(prepare_failed(format!(
                "only a serving engine prepares a generation, and this one is {}",
                name(self.lifecycle)
            )));
        }
        match self.generations.begin_prepare() {
            Ok(generation) => Step::Prepare { generation },
            Err(error) => Step::Report(prepare_failed(error.to_string())),
        }
    }

    /// A switch is accepted when it names the generation this engine prepared; the state moves
    /// with [`Self::switch`], once every device executes it.
    fn accept_switch(&self, generation: u64) -> Step {
        // "Prepared nothing" and "prepared another" are the same refusal to the daemon and two
        // different faults in the log, so the reason names which one.
        let refusal = match self.generations.update() {
            Generation::Prepared {
                generation: prepared,
            } if prepared == generation && self.lifecycle == Lifecycle::Serving => {
                return Step::Switch { generation };
            }
            Generation::Prepared { .. } if self.lifecycle != Lifecycle::Serving => format!(
                "a switch to generation {generation} of an engine that is {}: a generation is \
                 switched to at a batch boundary of a serving engine",
                name(self.lifecycle)
            ),
            Generation::Prepared {
                generation: prepared,
            } => format!(
                "a switch to generation {generation}, while this engine holds generation \
                 {prepared} prepared: the preparation of {generation} failed or never arrived"
            ),
            Generation::Preparing {
                generation: preparing,
            } => format!(
                "a switch to generation {generation}, while generation {preparing} is still \
                 being prepared"
            ),
            Generation::Steady | Generation::Trial { .. } => format!(
                "a switch to generation {generation}, which this engine prepared nothing for: \
                 the preparation failed or never arrived"
            ),
        };
        Step::Report(prepare_failed(refusal))
    }

    /// Every device executes `generation`: the trial begins, with the previous generation kept.
    pub(crate) fn switch(&mut self, generation: u64) -> worker::Result<()> {
        self.generations.switch(generation)
    }

    fn commit(&mut self) -> Step {
        match self.generations.update() {
            Generation::Trial { .. } => match self.generations.commit() {
                Ok(kept) => Step::Commit { kept },
                Err(error) => Step::Report(rolled_back(error.to_string())),
            },
            // The generation that executes is the committed one already: the daemon's commit is
            // acknowledged as often as it is sent.
            Generation::Steady => Step::Report(committed()),
            // Nothing is on trial, so nothing was committed, and the generation that executed
            // before the update still does.
            Generation::Preparing { generation } | Generation::Prepared { generation } => {
                Step::Report(rolled_back(format!(
                    "the commit names generation {generation}, which is not on trial: generation \
                     {} executes",
                    self.generations.current()
                )))
            }
        }
    }

    fn rollback(&mut self, reason: &str) -> Step {
        match self.generations.rollback() {
            Ok((left, executed)) => Step::Rollback { left, executed },
            // No update is in flight: the engine rolled back by itself, or prepared nothing. What
            // the daemon asks for already holds.
            Err(_) => Step::Report(rolled_back(reason.to_owned())),
        }
    }

    fn drain(&mut self, epoch: u64) -> Step {
        match self.lifecycle {
            Lifecycle::Serving => {
                self.lifecycle = Lifecycle::Draining {
                    serves_again: false,
                };
                // An update cannot cross the drain: a switch takes effect at a batch boundary
                // and a drained engine reaches none. A generation on trial stays as it is.
                let discarded = match self.generations.update() {
                    Generation::Preparing { .. } | Generation::Prepared { .. } => self
                        .generations
                        .rollback()
                        .ok()
                        .map(|(generation, _)| generation),
                    Generation::Steady | Generation::Trial { .. } => None,
                };
                Step::Drain { epoch, discarded }
            }
            // The drain in flight acknowledges.
            Lifecycle::Draining { .. } => {
                self.lifecycle = Lifecycle::Draining {
                    serves_again: false,
                };
                Step::Ignore
            }
            // A parked engine steps nothing and holds no request: it is drained, and hands
            // nothing over.
            Lifecycle::Parked => Step::Report(drained(epoch, Vec::new())),
            lifecycle @ (Lifecycle::Starting | Lifecycle::Stopped | Lifecycle::Failed) => {
                Step::Report(fault(
                    "drain",
                    format!("an engine that is {} cannot drain", name(lifecycle)),
                ))
            }
        }
    }

    fn promote(&mut self) -> Step {
        match self.lifecycle {
            Lifecycle::Parked => Step::Promote,
            // It serves already: the promotion is acknowledged as often as it is sent.
            Lifecycle::Serving => Step::Report(self.serving_report()),
            // The engine serves again once its schedulers parked: a promotion cannot overtake
            // the requests that are still ending.
            Lifecycle::Draining { .. } => {
                self.lifecycle = Lifecycle::Draining { serves_again: true };
                Step::Ignore
            }
            lifecycle @ (Lifecycle::Starting | Lifecycle::Stopped | Lifecycle::Failed) => {
                Step::Report(fault(
                    "promote",
                    format!(
                        "an engine that is {} cannot be promoted: only a parked engine is",
                        name(lifecycle)
                    ),
                ))
            }
        }
    }

    /// The start completed: the engine executes its first generation, parked when it was
    /// launched beside a serving engine and serving otherwise.
    pub(crate) fn started(&mut self, parked: bool) -> Report {
        if parked {
            self.lifecycle = Lifecycle::Parked;
            Report {
                kind: Some(report::Kind::Parked(proto::process::Parked {
                    launch_generation: self.launch_generation,
                    generation: self.generations.current(),
                })),
            }
        } else {
            self.lifecycle = Lifecycle::Serving;
            self.serving_report()
        }
    }

    /// The preparation of the update in flight finished.
    pub(crate) fn prepared(&mut self, generation: u64, uncompared: Vec<String>) -> Report {
        match self.generations.prepared(generation) {
            Ok(()) => Report {
                kind: Some(report::Kind::Prepared(proto::process::Prepared {
                    generation,
                    uncompared,
                })),
            },
            Err(error) => prepare_failed(format!(
                "the preparation of generation {generation} finished for an update that moved \
                 on: {error}"
            )),
        }
    }

    /// The preparation of the update in flight failed, and nothing changed.
    pub(crate) fn prepare_failed(&mut self, generation: u64, reason: String) -> Report {
        if let Err(error) = self.generations.prepare_failed(generation) {
            tracing::warn!(generation, %error, "a preparation failed for no update in flight");
        }
        prepare_failed(reason)
    }

    /// The captures of the switch to `generation` finished: it executes.
    pub(crate) fn switched(&self, generation: u64) -> Report {
        Report {
            kind: Some(report::Kind::Switched(proto::process::Switched {
                generation,
            })),
        }
    }

    /// The commit took effect.
    pub(crate) fn committed(&self) -> Report {
        committed()
    }

    /// The rollback took effect: the kept generation executes, for `reason`.
    pub(crate) fn rolled_back(&self, reason: String) -> Report {
        rolled_back(reason)
    }

    /// Rolls the update on trial back because of a failure the engine observed. Returns the
    /// generation left and whether it executed, as [`Step::Rollback`] carries them.
    pub(crate) fn rollback_on_failure(&mut self) -> worker::Result<(u64, bool)> {
        self.generations.rollback()
    }

    /// Every scheduler parked: the engine is drained and hands over `index`. Returns the
    /// acknowledgement, and whether a promotion that arrived meanwhile is due now.
    pub(crate) fn drained(
        &mut self,
        epoch: u64,
        index: Vec<proto::request::GroupIndex>,
    ) -> (Report, bool) {
        let serves_again = matches!(self.lifecycle, Lifecycle::Draining { serves_again: true });
        self.lifecycle = Lifecycle::Parked;
        (drained(epoch, index), serves_again)
    }

    /// The promotion took effect: the engine admits.
    pub(crate) fn promoted(&mut self) -> Report {
        self.lifecycle = Lifecycle::Serving;
        self.serving_report()
    }

    /// The engine was shut down.
    pub(crate) fn stop(&mut self) {
        self.lifecycle = Lifecycle::Stopped;
    }

    /// What `failure` means now: it abandons a preparation, rolls back a trial, or is the
    /// engine's own, which fails it.
    pub(crate) fn failure(&mut self, failure: &UpdateFailure) -> Action {
        let action = match self.lifecycle {
            // Nothing executed before a start's first generation, so nothing can go on.
            Lifecycle::Starting => Action::Fail,
            _ => self.generations.action(failure),
        };
        if action == Action::Fail {
            self.lifecycle = Lifecycle::Failed;
        }
        action
    }

    fn serving_report(&self) -> Report {
        Report {
            kind: Some(report::Kind::Serving(proto::process::Serving {
                launch_generation: self.launch_generation,
                generation: self.generations.current(),
            })),
        }
    }
}

/// The lifecycle state in a refusal's words.
fn name(lifecycle: Lifecycle) -> &'static str {
    match lifecycle {
        Lifecycle::Starting => "starting",
        Lifecycle::Parked => "parked",
        Lifecycle::Serving => "serving",
        Lifecycle::Draining { .. } => "draining",
        Lifecycle::Stopped => "stopped",
        Lifecycle::Failed => "failed",
    }
}

fn prepare_failed(reason: String) -> Report {
    Report {
        kind: Some(report::Kind::PrepareFailed(proto::process::PrepareFailed {
            reason,
        })),
    }
}

fn committed() -> Report {
    Report {
        kind: Some(report::Kind::Committed(proto::process::Committed {})),
    }
}

fn rolled_back(reason: String) -> Report {
    Report {
        kind: Some(report::Kind::RolledBack(proto::process::RolledBack {
            reason,
        })),
    }
}

fn drained(epoch: u64, index: Vec<proto::request::GroupIndex>) -> Report {
    Report {
        kind: Some(report::Kind::Drained(proto::process::Drained {
            epoch,
            // What the drain hands over is the index alone; whether the requests that migrate
            // recompute, and whether this engine's storage frees, are the daemon's to state to
            // the engine it promotes.
            handoff: Some(proto::process::Handoff {
                index,
                recompute: false,
                grows: false,
            }),
        })),
    }
}

/// A refusal of a lifecycle command, as the fault report the daemon reads.
pub(crate) fn fault(kind: &str, reason: String) -> Report {
    Report {
        kind: Some(report::Kind::Fault(proto::process::Fault {
            kind: kind.to_owned(),
            reason,
            device: 0,
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::process::Command;

    /// The launch generation the engines of these tests were started at.
    const LAUNCH: u64 = 7;

    fn starting() -> State {
        State::new(LAUNCH, Generations::new(Vec::new()))
    }

    /// An engine whose start completed with its first generation, parked or serving.
    fn started(parked: bool) -> State {
        let mut state = starting();
        state
            .generations
            .install_first(1)
            .expect("the first generation");
        state.started(parked);
        state
    }

    fn command(kind: command::Kind) -> Command {
        Command { kind: Some(kind) }
    }

    /// Sends `command` as the process interface delivers it.
    fn send(state: &mut State, command: Command) -> Step {
        state.command(&command.kind.expect("a command states its kind"))
    }

    fn prepare() -> Command {
        command(command::Kind::Prepare(proto::process::Prepare::default()))
    }

    fn switch(generation: u64) -> Command {
        command(command::Kind::Switch(proto::process::Switch { generation }))
    }

    fn commit() -> Command {
        command(command::Kind::Commit(proto::process::Commit {}))
    }

    fn rollback(reason: &str) -> Command {
        command(command::Kind::Rollback(proto::process::Rollback {
            reason: reason.to_owned(),
        }))
    }

    fn drain(epoch: u64) -> Command {
        command(command::Kind::Drain(proto::process::Drain { epoch }))
    }

    fn promote() -> Command {
        command(command::Kind::Promote(proto::process::Promote::default()))
    }

    fn kind(report: &Report) -> &report::Kind {
        report.kind.as_ref().expect("a report states its kind")
    }

    /// A serving engine whose update is on trial as generation 2.
    fn on_trial() -> State {
        let mut state = started(false);
        assert_eq!(send(&mut state, prepare()), Step::Prepare { generation: 2 });
        state.prepared(2, Vec::new());
        assert_eq!(send(&mut state, switch(2)), Step::Switch { generation: 2 });
        state.switch(2).expect("the devices switched");
        state
    }

    fn stall() -> UpdateFailure {
        UpdateFailure::Stalled {
            stall: ipc::Stall {
                batch_id: Some(3),
                route: None,
                method: None,
                calls: Vec::new(),
                running_ms: 20_000,
            },
        }
    }

    #[test]
    fn an_engine_launched_to_serve_starts_serving_its_first_generation() {
        let mut state = starting();
        assert_eq!(state.lifecycle().state(), EngineState::Starting);
        assert!(state.lifecycle().refusal().is_some());
        state
            .generations
            .install_first(1)
            .expect("the first generation");

        let report = state.started(false);
        assert_eq!(
            kind(&report),
            &report::Kind::Serving(proto::process::Serving {
                launch_generation: LAUNCH,
                generation: 1,
            })
        );
        assert_eq!(state.lifecycle().state(), EngineState::Serving);
        assert!(state.lifecycle().refusal().is_none());
        // Startup is not an update: nothing is kept and nothing is on trial.
        assert_eq!(state.update(), Generation::Steady);
    }

    #[test]
    fn an_engine_launched_beside_a_serving_one_parks_and_serves_when_promoted() {
        let mut state = starting();
        state
            .generations
            .install_first(1)
            .expect("the first generation");
        let report = state.started(true);
        assert_eq!(
            kind(&report),
            &report::Kind::Parked(proto::process::Parked {
                launch_generation: LAUNCH,
                generation: 1,
            })
        );
        assert_eq!(state.lifecycle().state(), EngineState::Parked);
        assert!(state.lifecycle().refusal().is_some());

        assert_eq!(send(&mut state, promote()), Step::Promote);
        let report = state.promoted();
        assert_eq!(
            kind(&report),
            &report::Kind::Serving(proto::process::Serving {
                launch_generation: LAUNCH,
                generation: 1,
            })
        );
        assert_eq!(state.lifecycle().state(), EngineState::Serving);
    }

    #[test]
    fn a_promotion_of_a_serving_engine_is_acknowledged_again() {
        let mut state = started(false);
        let Step::Report(report) = send(&mut state, promote()) else {
            panic!("a serving engine has nothing to promote");
        };
        assert!(matches!(kind(&report), report::Kind::Serving(_)));
        assert_eq!(state.lifecycle(), Lifecycle::Serving);
    }

    #[test]
    fn a_starting_engine_refuses_the_lifecycle_commands() {
        let mut state = starting();
        for refused in [promote(), drain(4)] {
            let Step::Report(report) = send(&mut state, refused) else {
                panic!("a starting engine took a lifecycle command");
            };
            assert!(matches!(kind(&report), report::Kind::Fault(_)));
            assert_eq!(state.lifecycle(), Lifecycle::Starting);
        }
    }

    #[test]
    fn a_drain_parks_the_engine_and_hands_its_index_over() {
        let mut state = started(false);
        assert_eq!(
            send(&mut state, drain(4)),
            Step::Drain {
                epoch: 4,
                discarded: None
            }
        );
        assert_eq!(state.lifecycle().state(), EngineState::Draining);
        assert!(state.lifecycle().refusal().is_some());

        let index = vec![proto::request::GroupIndex {
            worker_group: "model".to_owned(),
            ..Default::default()
        }];
        let (report, serves_again) = state.drained(4, index.clone());
        assert!(!serves_again);
        let report::Kind::Drained(drained) = kind(&report) else {
            panic!("a drain is acknowledged with Drained");
        };
        assert_eq!(drained.epoch, 4);
        assert_eq!(
            drained.handoff.as_ref().map(|handoff| &handoff.index),
            Some(&index)
        );
        assert_eq!(state.lifecycle().state(), EngineState::Parked);
    }

    #[test]
    fn a_drained_engine_serves_again_when_promoted() {
        let mut state = started(false);
        send(&mut state, drain(4));
        state.drained(4, Vec::new());
        assert_eq!(send(&mut state, promote()), Step::Promote);
        assert!(matches!(kind(&state.promoted()), report::Kind::Serving(_)));
        assert_eq!(state.lifecycle(), Lifecycle::Serving);
    }

    #[test]
    fn a_promotion_that_overtakes_a_drain_takes_effect_once_the_engine_parked() {
        let mut state = started(false);
        send(&mut state, drain(4));
        assert_eq!(send(&mut state, promote()), Step::Ignore);
        assert_eq!(state.lifecycle().state(), EngineState::Draining);
        let (report, serves_again) = state.drained(4, Vec::new());
        assert!(matches!(kind(&report), report::Kind::Drained(_)));
        assert!(serves_again);
    }

    #[test]
    fn a_drain_of_a_parked_engine_is_acknowledged_with_nothing_handed_over() {
        let mut state = started(true);
        let Step::Report(report) = send(&mut state, drain(9)) else {
            panic!("a parked engine has nothing to drain");
        };
        let report::Kind::Drained(drained) = kind(&report) else {
            panic!("a drain is acknowledged with Drained");
        };
        assert_eq!(drained.epoch, 9);
        assert!(
            drained
                .handoff
                .as_ref()
                .is_some_and(|handoff| handoff.index.is_empty())
        );
        assert_eq!(state.lifecycle(), Lifecycle::Parked);
    }

    #[test]
    fn a_generation_update_is_prepared_switched_to_and_committed() {
        let mut state = started(false);
        assert_eq!(send(&mut state, prepare()), Step::Prepare { generation: 2 });
        assert_eq!(state.update(), Generation::Preparing { generation: 2 });

        let report = state.prepared(2, vec!["rms_norm: no shadow output".to_owned()]);
        assert_eq!(
            kind(&report),
            &report::Kind::Prepared(proto::process::Prepared {
                generation: 2,
                uncompared: vec!["rms_norm: no shadow output".to_owned()],
            })
        );
        assert_eq!(state.update(), Generation::Prepared { generation: 2 });

        assert_eq!(send(&mut state, switch(2)), Step::Switch { generation: 2 });
        state.switch(2).expect("the devices switched");
        assert_eq!(state.update(), Generation::Trial { generation: 2 });
        assert_eq!(
            kind(&state.switched(2)),
            &report::Kind::Switched(proto::process::Switched { generation: 2 })
        );

        assert_eq!(send(&mut state, commit()), Step::Commit { kept: 1 });
        assert!(matches!(
            kind(&state.committed()),
            report::Kind::Committed(_)
        ));
        assert_eq!(state.update(), Generation::Steady);
        // The engine went on serving throughout.
        assert_eq!(state.lifecycle(), Lifecycle::Serving);
    }

    #[test]
    fn a_failed_preparation_changes_nothing() {
        let mut state = started(false);
        send(&mut state, prepare());
        let report = state.prepare_failed(2, "a module did not resolve".to_owned());
        assert_eq!(
            kind(&report),
            &report::Kind::PrepareFailed(proto::process::PrepareFailed {
                reason: "a module did not resolve".to_owned(),
            })
        );
        assert_eq!(state.update(), Generation::Steady);
        assert_eq!(state.lifecycle(), Lifecycle::Serving);
        // The switch the daemon may still send finds nothing prepared.
        let Step::Report(report) = send(&mut state, switch(2)) else {
            panic!("a switch to nothing prepared was accepted");
        };
        assert!(matches!(kind(&report), report::Kind::PrepareFailed(_)));
    }

    #[test]
    fn a_prepare_is_refused_while_an_update_is_in_flight_or_the_engine_does_not_serve() {
        let mut state = started(false);
        send(&mut state, prepare());
        let Step::Report(report) = send(&mut state, prepare()) else {
            panic!("a second update began");
        };
        assert!(matches!(kind(&report), report::Kind::PrepareFailed(_)));

        let mut parked = started(true);
        let Step::Report(report) = send(&mut parked, prepare()) else {
            panic!("a parked engine prepared a generation");
        };
        assert!(matches!(kind(&report), report::Kind::PrepareFailed(_)));
        assert_eq!(parked.update(), Generation::Steady);
    }

    #[test]
    fn a_switch_names_the_generation_that_was_prepared() {
        let mut state = started(false);
        send(&mut state, prepare());
        // Still preparing.
        assert!(matches!(send(&mut state, switch(2)), Step::Report(_)));
        state.prepared(2, Vec::new());
        // Another number than the one prepared.
        assert!(matches!(send(&mut state, switch(3)), Step::Report(_)));
        assert_eq!(state.update(), Generation::Prepared { generation: 2 });
        assert_eq!(send(&mut state, switch(2)), Step::Switch { generation: 2 });
    }

    #[test]
    fn a_rollback_by_decision_returns_to_the_kept_generation() {
        let mut state = on_trial();
        assert_eq!(
            send(&mut state, rollback("the operator rolled the trial back")),
            Step::Rollback {
                left: 2,
                executed: true
            }
        );
        assert_eq!(state.update(), Generation::Steady);
        assert_eq!(
            kind(&state.rolled_back("the operator rolled the trial back".to_owned())),
            &report::Kind::RolledBack(proto::process::RolledBack {
                reason: "the operator rolled the trial back".to_owned(),
            })
        );
        // The engine executes its first generation again, which its next report states.
        let Step::Report(report) = send(&mut state, promote()) else {
            panic!("a serving engine has nothing to promote");
        };
        assert_eq!(
            kind(&report),
            &report::Kind::Serving(proto::process::Serving {
                launch_generation: LAUNCH,
                generation: 1,
            })
        );
    }

    #[test]
    fn a_rollback_before_the_switch_discards_the_update() {
        // While it is being prepared, and once it is prepared.
        for prepared in [false, true] {
            let mut state = started(false);
            send(&mut state, prepare());
            if prepared {
                state.prepared(2, Vec::new());
            }
            assert_eq!(
                send(&mut state, rollback("abandoned")),
                Step::Rollback {
                    left: 2,
                    executed: false
                }
            );
            assert_eq!(state.update(), Generation::Steady);
        }
    }

    #[test]
    fn a_commit_or_a_rollback_with_no_update_in_flight_is_acknowledged() {
        let mut state = started(false);
        let Step::Report(report) = send(&mut state, commit()) else {
            panic!("a commit of nothing started work");
        };
        assert!(matches!(kind(&report), report::Kind::Committed(_)));

        let Step::Report(report) = send(&mut state, rollback("a peer failed")) else {
            panic!("a rollback of nothing started work");
        };
        assert_eq!(
            kind(&report),
            &report::Kind::RolledBack(proto::process::RolledBack {
                reason: "a peer failed".to_owned(),
            })
        );
        assert_eq!(state.update(), Generation::Steady);
    }

    #[test]
    fn a_commit_before_the_switch_commits_nothing() {
        let mut state = started(false);
        send(&mut state, prepare());
        state.prepared(2, Vec::new());
        let Step::Report(report) = send(&mut state, commit()) else {
            panic!("a commit of a generation that never executed started work");
        };
        assert!(matches!(kind(&report), report::Kind::RolledBack(_)));
        // The prepared generation still waits for its switch.
        assert_eq!(state.update(), Generation::Prepared { generation: 2 });
    }

    #[test]
    fn a_fault_on_trial_rolls_back_and_the_engine_serves_on() {
        let mut state = on_trial();
        assert_eq!(state.failure(&stall()), Action::Rollback);
        assert_eq!(
            state.rollback_on_failure().expect("the trial rolls back"),
            (2, true)
        );
        assert_eq!(state.update(), Generation::Steady);
        assert_eq!(state.lifecycle(), Lifecycle::Serving);
        // The daemon's own rollback, sent when it reads the fault, is acknowledged too.
        assert!(matches!(
            send(&mut state, rollback("a stalled job")),
            Step::Report(_)
        ));
    }

    #[test]
    fn every_failure_that_is_not_process_fatal_rolls_a_trial_back() {
        let failures = [
            stall(),
            UpdateFailure::NonFiniteLogits {
                request: ipc::RequestKey::new(0, ipc::RequestId(5), 0),
            },
            UpdateFailure::Policy {
                error: "the policy trapped".to_owned(),
            },
            UpdateFailure::Device {
                device: 1,
                error: "a capture failed".to_owned(),
                batch: None,
            },
            UpdateFailure::Threshold {
                reason: "the step time regressed".to_owned(),
            },
        ];
        for failure in failures {
            let mut state = on_trial();
            assert_eq!(state.failure(&failure), Action::Rollback, "{failure:?}");
            assert_eq!(
                state.rollback_on_failure().expect("the trial rolls back"),
                (2, true)
            );
            assert_eq!(
                kind(&state.rolled_back(failure.reason())),
                &report::Kind::RolledBack(proto::process::RolledBack {
                    reason: failure.reason(),
                })
            );
            assert_eq!(state.lifecycle(), Lifecycle::Serving);
            assert_eq!(state.update(), Generation::Steady);
        }
    }

    #[test]
    fn a_fault_outside_a_trial_fails_the_engine() {
        let mut state = started(false);
        assert_eq!(state.failure(&stall()), Action::Fail);
        assert_eq!(state.lifecycle().state(), EngineState::Failed);
        assert!(state.lifecycle().refusal().is_some());

        // A start that fails is the engine's failure whatever failed.
        let mut state = starting();
        let failure = UpdateFailure::Prepare {
            reason: "a module did not resolve".to_owned(),
        };
        assert_eq!(state.failure(&failure), Action::Fail);
        assert_eq!(state.lifecycle(), Lifecycle::Failed);
    }

    #[test]
    fn a_failed_preparation_is_abandoned_and_fails_nothing() {
        let mut state = started(false);
        send(&mut state, prepare());
        let failure = UpdateFailure::Prepare {
            reason: "a shadow batch was out of tolerance".to_owned(),
        };
        assert_eq!(state.failure(&failure), Action::Abandon);
        assert_eq!(state.lifecycle(), Lifecycle::Serving);
    }

    #[test]
    fn a_drain_discards_an_update_that_was_not_switched_to() {
        let mut state = started(false);
        send(&mut state, prepare());
        state.prepared(2, Vec::new());
        assert_eq!(
            send(&mut state, drain(4)),
            Step::Drain {
                epoch: 4,
                discarded: Some(2)
            }
        );
        assert_eq!(state.update(), Generation::Steady);

        // A generation on trial stays on trial across the drain.
        let mut state = on_trial();
        assert_eq!(
            send(&mut state, drain(4)),
            Step::Drain {
                epoch: 4,
                discarded: None
            }
        );
        assert_eq!(state.update(), Generation::Trial { generation: 2 });
    }

    #[test]
    fn a_shutdown_stops_the_engine() {
        let mut state = started(false);
        state.stop();
        assert_eq!(state.lifecycle().state(), EngineState::Stopped);
        assert!(state.lifecycle().refusal().is_some());
    }

    #[test]
    fn a_pause_and_a_frontends_command_move_nothing() {
        let mut state = started(false);
        assert_eq!(
            send(
                &mut state,
                command(command::Kind::Pause(proto::process::Pause { paused: true }))
            ),
            Step::Pause(true)
        );
        assert_eq!(
            send(
                &mut state,
                command(command::Kind::Serve(proto::process::Serve::default()))
            ),
            Step::Ignore
        );
        assert_eq!(state.lifecycle(), Lifecycle::Serving);
        assert_eq!(state.update(), Generation::Steady);
    }
}
