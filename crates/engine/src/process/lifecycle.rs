//! Where an engine is: its lifecycle and, while it serves, its generation update (`specs/updates.md`
//! §3.3), as rules with no I/O.
//!
//! [`Lifecycle`] is where the engine is in its life and [`Update`] where a serving engine's
//! generation update is. Both are moved only by messages: a command of the daemon's process
//! interface, the end of the work a command started, or a failure the engine observed. Each
//! function below states what one message does in each state: it moves the enums and returns
//! what the control thread does next, or the report that answers the command at once (an
//! acknowledgement that needs no work, or a refusal). Nothing here touches a device, a scheduler
//! or a socket.
//!
//! | From | Message | To | Report |
//! | --- | --- | --- | --- |
//! | `Starting` | the start ended, launched beside a serving engine | `Parked` | `Parked` |
//! | `Starting` | the start ended | `Serving` | `Serving` |
//! | `Parked` | `Promote` | `Serving`, once its regions took the epoch's sizes and admission opened; a kept grow applies | `Serving` |
//! | `Serving` | `Promote` | `Serving` | `Serving` again |
//! | `Serving` | `Drain` | `Draining` | — |
//! | `Draining` | `Drain` | `Draining`, the drain in flight answers | — |
//! | `Draining` | every scheduler drained | `Parked` | `Drained` |
//! | `Draining` | `Promote` | `Draining`, then `Serving` once parked | `Drained`, `Serving` |
//! | `Parked` | `Drain` | `Parked` | `Drained` at once |
//! | `Starting`, `Stopped`, `Failed` | `Drain`, `Promote` | unchanged | `Fault` |
//! | `Serving` | `Grow`, a promotion's `grow` | unchanged; the regions grow now | — |
//! | `Starting`, `Parked`, `Draining` | `Grow`, a promotion's `grow` | unchanged; the grow is kept | — |
//! | `Starting` | the start ended | `Parked`, a kept grow kept; or `Serving`, a kept grow applying | as above |
//! | `Draining` | every scheduler drained | `Serving` again on a promotion, a kept grow applying; `Parked` otherwise, a kept grow dropped | `Drained` |
//! | any | shutdown | `Stopped` | — |
//! | any | a process-fatal failure | `Failed` | the failure's own |
//!
//! | From | Message | To | Report |
//! | --- | --- | --- | --- |
//! | `Steady`, serving | `Prepare` | `Preparing(n)`, `n` one past the generation that executes | — |
//! | any other | `Prepare` | unchanged | `PrepareFailed` |
//! | `Preparing(n)` | the preparation ended | `Prepared(n)` | `Prepared` |
//! | `Preparing(n)` | the preparation failed | `Steady` | `PrepareFailed` |
//! | `Prepared(n)`, serving | `Switch(n)` | `Trial(n)`; the switch is queued at a batch boundary | `Switched` once it ran |
//! | any other | `Switch` | unchanged | `PrepareFailed` |
//! | `Trial(n)` | the switch failed | `Steady`, the previous generation executing | `RolledBack` |
//! | `Trial(n)` | `Commit` | `Steady` | `Committed` |
//! | `Steady` | `Commit` | `Steady` | `Committed` again |
//! | `Preparing(n)`, `Prepared(n)` | `Commit` | unchanged | `RolledBack`: nothing is on trial |
//! | `Trial(n)`, not draining | `Rollback`, or a failure that is not process-fatal | `Steady`, on the kept generation | `RolledBack` |
//! | `Trial(n)`, draining | `Rollback` | unchanged | `Fault` |
//! | `Trial(n)`, draining | a failure that is not process-fatal | unchanged | the failure's own |
//! | `Preparing(n)`, `Prepared(n)` | `Rollback`, `Drain` | `Steady`, the update discarded | `RolledBack` |
//! | `Steady` | `Rollback` | `Steady` | `RolledBack` again |
//!
//! A switch moves the update to `Trial` when it is accepted rather than when it ran: it runs on
//! the operations thread, which serializes every job that pauses the schedulers, so a commit or a
//! rollback accepted after it runs after it.
//!
//! A scheduler's `Resume` ends its drain as well as its pause, so no job that pauses the
//! schedulers starts while the engine drains ([`Lifecycle::boundary_refusal`]): the drain itself
//! begins on the operations thread, behind every such job accepted before it, and a trial's
//! switch back waits for the daemon once the engine is drained.

use proto::process::{Report, report};
use proto::request::wire::EngineState;

/// Where an engine is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lifecycle {
    /// It loads its ranks, opens its session, resolves its modules and captures its graphs.
    Starting,
    /// It executes nothing and admits nothing, with its storage mapped and its graphs captured:
    /// started beside a serving engine (a switch's target, a standby), or drained.
    Parked,
    /// It admits and executes requests.
    Serving,
    /// It admits nothing more, and steps its requests in flight until each ends by itself or the
    /// drain's deadline ends it as migratable; parked once every scheduler drained.
    Draining {
        /// A `Promote` arrived meanwhile: the engine serves again once it is parked.
        serves_again: bool,
    },
    /// It was shut down.
    Stopped,
    /// It failed and serves nothing more.
    Failed,
}

/// Where a serving engine's generation update is; each state names the generation it concerns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Update {
    /// One generation executes and no update is in flight.
    Steady,
    /// The generation is being resolved, installed, rebound or recaptured, and shadowed, beside
    /// the one that executes.
    Preparing(u64),
    /// The generation is ready for its switch; the previous one still executes.
    Prepared(u64),
    /// The generation executes, or its switch is queued, and the previous one is kept.
    Trial(u64),
}

impl Lifecycle {
    /// The state as every request front announces it.
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

    /// Why a job that pauses the schedulers at a batch boundary is refused in this state;
    /// `None` when it may run. A `Resume` ends a scheduler's drain in flight as well as its pause,
    /// so none starts while the engine drains.
    pub(crate) fn boundary_refusal(self) -> Option<&'static str> {
        matches!(self, Self::Draining { .. }).then_some(
            "the engine drains, and a pause of its schedulers would end the drain: the operation \
             runs once the engine is drained",
        )
    }

    /// The state in a refusal's words.
    fn name(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Parked => "parked",
            Self::Serving => "serving",
            Self::Draining { .. } => "draining",
            Self::Stopped => "stopped",
            Self::Failed => "failed",
        }
    }
}

/// The lifecycle a start that ended reaches: parked when the launch names the engine it
/// succeeds, which opens admission only when promoted; otherwise serving, once admission opened.
pub(crate) fn started(parked: bool) -> Lifecycle {
    if parked {
        Lifecycle::Parked
    } else {
        Lifecycle::Serving
    }
}

/// `Drain { epoch }`: `Ok(true)` when the control thread begins a drain (the engine served and
/// drains now), `Ok(false)` when the drain in flight answers it, and the report that answers it at
/// once otherwise: a parked engine holds no request and is drained already.
pub(crate) fn drain(lifecycle: &mut Lifecycle, epoch: u64) -> Result<bool, Report> {
    match *lifecycle {
        Lifecycle::Serving => {
            *lifecycle = Lifecycle::Draining {
                serves_again: false,
            };
            Ok(true)
        }
        // A promotion that arrived before is overtaken: the drain asked for last stands.
        Lifecycle::Draining { .. } => {
            *lifecycle = Lifecycle::Draining {
                serves_again: false,
            };
            Ok(false)
        }
        Lifecycle::Parked => Err(drained_report(epoch)),
        state @ (Lifecycle::Starting | Lifecycle::Stopped | Lifecycle::Failed) => Err(fault(
            proto::process::FaultKind::Drain,
            format!("an engine that is {} cannot drain", state.name()),
            None,
        )),
    }
}

/// Every scheduler drained: the engine is parked. Returns whether a promotion that arrived
/// meanwhile makes it serve again now. A grow `kept` during the drain applies only then: an
/// engine that stays parked gave its memory to the epoch that serves, and drops it.
pub(crate) fn drained(lifecycle: &mut Lifecycle, kept: &mut bool) -> bool {
    let serves_again = matches!(lifecycle, Lifecycle::Draining { serves_again: true });
    *lifecycle = Lifecycle::Parked;
    if !serves_again {
        *kept = false;
    }
    serves_again
}

/// `Grow`, or a promotion's `grow` (`specs/updates.md` §3.7): `true` when the control thread
/// grows the regions now, the engine serving. Otherwise the grow is `kept` while the engine
/// starts, whose session and regions do not exist yet, while it is parked, whose regions keep
/// what the buffer pool holds until its promotion resizes them (`epoch::promote`; a target of the
/// serving epoch's layout shares them with the serving engine), and while it drains, whose
/// regions shrink; it applies once a promotion makes the engine serve, one that overtook a drain
/// included, and a drain that parks the engine drops it ([`drained`]). A stopped or failed
/// engine grows nothing.
pub(crate) fn grow(lifecycle: Lifecycle, kept: &mut bool) -> bool {
    match lifecycle {
        Lifecycle::Serving => true,
        Lifecycle::Starting | Lifecycle::Parked | Lifecycle::Draining { .. } => {
            *kept = true;
            false
        }
        Lifecycle::Stopped | Lifecycle::Failed => false,
    }
}

/// `Promote`: `Ok(true)` when the control thread opens admission now (the engine is parked),
/// `Ok(false)` when it does so once a drain in flight parked it, and the report that answers it at
/// once otherwise: `serving`, the acknowledgement of an engine that serves already, or a refusal.
pub(crate) fn promote(lifecycle: &mut Lifecycle, serving: Report) -> Result<bool, Report> {
    match *lifecycle {
        Lifecycle::Parked => Ok(true),
        Lifecycle::Serving => Err(serving),
        // A promotion cannot overtake the requests that are still ending.
        Lifecycle::Draining { .. } => {
            *lifecycle = Lifecycle::Draining { serves_again: true };
            Ok(false)
        }
        state @ (Lifecycle::Starting | Lifecycle::Stopped | Lifecycle::Failed) => Err(fault(
            proto::process::FaultKind::Promote,
            format!(
                "an engine that is {} cannot be promoted: only a parked engine is",
                state.name()
            ),
            None,
        )),
    }
}

/// `Prepare`: the number of the generation to prepare, one past `executing`, the generation
/// that executes; or the `PrepareFailed` that refuses it, changing nothing.
pub(crate) fn prepare(
    lifecycle: Lifecycle,
    update: &mut Update,
    executing: u64,
) -> Result<u64, Report> {
    if lifecycle != Lifecycle::Serving {
        return Err(prepare_failed(format!(
            "only a serving engine prepares a generation, and this one is {}",
            lifecycle.name()
        )));
    }
    if *update != Update::Steady {
        return Err(prepare_failed(format!(
            "a generation update is already in flight ({update:?})"
        )));
    }

    let generation = executing + 1;
    *update = Update::Preparing(generation);
    Ok(generation)
}

/// The preparation of `generation` ended: true when the update still waits for it, which is then
/// prepared; false when a rollback or a drain discarded it meanwhile, and what it built is the
/// caller's to discard.
pub(crate) fn prepared(update: &mut Update, generation: u64) -> bool {
    if *update != Update::Preparing(generation) {
        return false;
    }
    *update = Update::Prepared(generation);
    true
}

/// The preparation of `generation` failed: nothing changed, and the update ends if it still
/// waited for it.
pub(crate) fn prepare_ended(update: &mut Update, generation: u64) {
    if *update == Update::Preparing(generation) {
        *update = Update::Steady;
    }
}

/// `Switch { generation }`: accepted when it names the prepared generation of a serving engine,
/// which goes on trial; otherwise the `PrepareFailed` that refuses it, changing nothing.
pub(crate) fn switch(
    lifecycle: Lifecycle,
    update: &mut Update,
    generation: u64,
) -> Result<(), Report> {
    // "Prepared nothing" and "prepared another" are one refusal to the daemon and two faults in
    // the log, so the reason names which.
    let refusal = match *update {
        Update::Prepared(prepared) if prepared == generation && lifecycle == Lifecycle::Serving => {
            *update = Update::Trial(generation);
            return Ok(());
        }
        Update::Prepared(_) if lifecycle != Lifecycle::Serving => format!(
            "a switch to generation {generation} of an engine that is {}: a generation is \
             switched to at a batch boundary of a serving engine",
            lifecycle.name()
        ),
        Update::Prepared(prepared) => format!(
            "a switch to generation {generation}, while this engine holds generation {prepared} \
             prepared"
        ),
        Update::Preparing(preparing) => format!(
            "a switch to generation {generation}, while generation {preparing} is still being \
             prepared"
        ),
        Update::Steady | Update::Trial(_) => format!(
            "a switch to generation {generation}, which this engine prepared nothing for: the \
             preparation failed or never arrived"
        ),
    };
    Err(prepare_failed(refusal))
}

/// The switch to `generation` failed: the previous generation executes again and the update
/// ends. False when the update had moved on, which then holds as it is.
pub(crate) fn switch_failed(update: &mut Update, generation: u64) -> bool {
    if *update != Update::Trial(generation) {
        return false;
    }
    *update = Update::Steady;
    true
}

/// `Commit`: the generation on trial, which is committed; otherwise the report that answers at
/// once: `Committed` again when nothing is on trial and nothing is in flight, and a `RolledBack`
/// stating that nothing is on trial while a generation is prepared or being prepared, which waits
/// on for its switch.
pub(crate) fn commit(update: &mut Update, executing: u64) -> Result<u64, Report> {
    match *update {
        Update::Trial(generation) => {
            *update = Update::Steady;
            Ok(generation)
        }
        Update::Steady => Err(committed()),
        Update::Preparing(generation) | Update::Prepared(generation) => Err(rolled_back(format!(
            "the commit names generation {generation}, which is not on trial: generation \
             {executing} executes"
        ))),
    }
}

/// `Rollback` of an engine in `lifecycle`: the generation left, and whether it executed (a switch
/// back at a batch boundary) or was only prepared (a discard); otherwise the report that answers
/// at once: the `RolledBack` for `reason` when no update is in flight and what the command asks
/// for holds already, or the `Fault` that refuses the switch back of a trial while the engine
/// drains, which leaves the trial as it is.
pub(crate) fn rollback(
    lifecycle: Lifecycle,
    update: &mut Update,
    reason: &str,
) -> Result<(u64, bool), Report> {
    let left = match *update {
        Update::Trial(generation) => {
            if let Some(refusal) = lifecycle.boundary_refusal() {
                return Err(fault(
                    proto::process::FaultKind::Rollback,
                    format!("generation {generation} stays on trial: {refusal}"),
                    None,
                ));
            }
            (generation, true)
        }
        Update::Preparing(generation) | Update::Prepared(generation) => (generation, false),
        Update::Steady => return Err(rolled_back(reason.to_owned())),
    };
    *update = Update::Steady;
    Ok(left)
}

/// A failure that is not process-fatal (non-finite logits, a policy fault) in an engine in
/// `lifecycle`: the generation on trial, which rolls back by itself; `None` outside a trial, and
/// while the engine drains, where the failure is reported and changes nothing.
pub(crate) fn fault_on_trial(lifecycle: Lifecycle, update: &mut Update) -> Option<u64> {
    let Update::Trial(generation) = *update else {
        return None;
    };
    if lifecycle.boundary_refusal().is_some() {
        return None;
    }
    *update = Update::Steady;
    Some(generation)
}

/// A drain began: an update that no switch took any more is discarded, and its generation
/// returned; a generation on trial stays on trial.
pub(crate) fn discarded_by_drain(update: &mut Update) -> Option<u64> {
    match *update {
        Update::Preparing(generation) | Update::Prepared(generation) => {
            *update = Update::Steady;
            Some(generation)
        }
        Update::Steady | Update::Trial(_) => None,
    }
}

/// A report of `kind`.
pub(crate) fn report(kind: report::Kind) -> Report {
    Report { kind: Some(kind) }
}

/// `Serving` at `generation`, for an engine launched at `launch_generation`.
pub(crate) fn serving(launch_generation: u64, generation: u64) -> Report {
    report(report::Kind::Serving(proto::process::Serving {
        launch_generation,
        generation,
    }))
}

/// `Parked` at `generation`, for an engine launched at `launch_generation`.
pub(crate) fn parked(launch_generation: u64, generation: u64) -> Report {
    report(report::Kind::Parked(proto::process::Parked {
        launch_generation,
        generation,
    }))
}

/// `Drained` of `epoch`.
pub(crate) fn drained_report(epoch: u64) -> Report {
    report(report::Kind::Drained(proto::process::Drained { epoch }))
}

/// `PrepareFailed` for `reason`.
pub(crate) fn prepare_failed(reason: String) -> Report {
    report(report::Kind::PrepareFailed(proto::process::PrepareFailed {
        reason,
    }))
}

/// `Committed`.
pub(crate) fn committed() -> Report {
    report(report::Kind::Committed(proto::process::Committed {}))
}

/// `RolledBack` for `reason`.
pub(crate) fn rolled_back(reason: String) -> Report {
    report(report::Kind::RolledBack(proto::process::RolledBack {
        reason,
    }))
}

/// A `Fault` of `kind`, on device `device` when it names one.
pub(crate) fn fault(
    kind: proto::process::FaultKind,
    reason: String,
    device: Option<u32>,
) -> Report {
    report(report::Kind::Fault(proto::process::Fault {
        kind: kind.into(),
        reason,
        device,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The launch generation and the generation of the engines of these tests.
    const LAUNCH: u64 = 7;

    fn kind(report: &Report) -> &report::Kind {
        report.kind.as_ref().expect("a report states its kind")
    }

    /// A serving engine whose generation 2 is on trial.
    fn on_trial() -> Update {
        let mut update = Update::Steady;
        assert_eq!(prepare(Lifecycle::Serving, &mut update, 1), Ok(2));
        assert!(prepared(&mut update, 2));
        assert_eq!(switch(Lifecycle::Serving, &mut update, 2), Ok(()));
        update
    }

    #[test]
    fn a_start_ends_serving_or_parked_as_its_launch_states() {
        assert_eq!(started(false), Lifecycle::Serving);
        assert_eq!(started(false).state(), EngineState::Serving);
        assert!(started(false).refusal().is_none());

        assert_eq!(started(true), Lifecycle::Parked);
        assert_eq!(started(true).state(), EngineState::Parked);
        assert!(started(true).refusal().is_some());
    }

    #[test]
    fn only_a_serving_engine_admits() {
        for refusing in [
            Lifecycle::Starting,
            Lifecycle::Parked,
            Lifecycle::Draining {
                serves_again: false,
            },
            Lifecycle::Stopped,
            Lifecycle::Failed,
        ] {
            assert!(refusing.refusal().is_some(), "{refusing:?} admitted");
        }
        assert_eq!(Lifecycle::Starting.state(), EngineState::Starting);
        assert_eq!(
            Lifecycle::Draining { serves_again: true }.state(),
            EngineState::Draining
        );
        assert_eq!(Lifecycle::Stopped.state(), EngineState::Stopped);
        assert_eq!(Lifecycle::Failed.state(), EngineState::Failed);
    }

    #[test]
    fn a_drain_of_a_serving_engine_drains_and_parks_it() {
        let mut lifecycle = Lifecycle::Serving;
        assert_eq!(drain(&mut lifecycle, 4), Ok(true));
        assert_eq!(lifecycle.state(), EngineState::Draining);
        assert!(lifecycle.refusal().is_some());

        assert!(!drained(&mut lifecycle, &mut false));
        assert_eq!(lifecycle, Lifecycle::Parked);
    }

    #[test]
    fn a_second_drain_is_answered_by_the_drain_in_flight() {
        let mut lifecycle = Lifecycle::Serving;
        assert_eq!(drain(&mut lifecycle, 4), Ok(true));
        assert_eq!(drain(&mut lifecycle, 4), Ok(false));
        assert_eq!(
            lifecycle,
            Lifecycle::Draining {
                serves_again: false
            }
        );
    }

    #[test]
    fn a_parked_engine_answers_a_drain_at_once() {
        let mut lifecycle = Lifecycle::Parked;
        let report = drain(&mut lifecycle, 9).expect_err("nothing to drain");
        assert_eq!(
            kind(&report),
            &report::Kind::Drained(proto::process::Drained { epoch: 9 })
        );
        assert_eq!(lifecycle, Lifecycle::Parked);
    }

    #[test]
    fn a_promotion_of_a_parked_engine_opens_admission() {
        let mut lifecycle = Lifecycle::Parked;
        assert_eq!(promote(&mut lifecycle, serving(LAUNCH, 1)), Ok(true));
        // The engine serves once admission opened, which the control thread states.
        assert_eq!(lifecycle, Lifecycle::Parked);
    }

    #[test]
    fn a_promotion_of_a_serving_engine_is_acknowledged_again() {
        let mut lifecycle = Lifecycle::Serving;
        let report = promote(&mut lifecycle, serving(LAUNCH, 3)).expect_err("serves already");
        assert_eq!(
            kind(&report),
            &report::Kind::Serving(proto::process::Serving {
                launch_generation: LAUNCH,
                generation: 3,
            })
        );
        assert_eq!(lifecycle, Lifecycle::Serving);
    }

    #[test]
    fn a_promotion_that_overtakes_a_drain_takes_effect_once_parked() {
        let mut lifecycle = Lifecycle::Serving;
        assert_eq!(drain(&mut lifecycle, 4), Ok(true));
        assert_eq!(promote(&mut lifecycle, serving(LAUNCH, 1)), Ok(false));
        assert_eq!(lifecycle.state(), EngineState::Draining);

        assert!(drained(&mut lifecycle, &mut false));
        assert_eq!(lifecycle, Lifecycle::Parked);
    }

    #[test]
    fn a_drain_after_a_promotion_overtook_one_keeps_the_engine_parked() {
        let mut lifecycle = Lifecycle::Serving;
        assert_eq!(drain(&mut lifecycle, 4), Ok(true));
        assert_eq!(promote(&mut lifecycle, serving(LAUNCH, 1)), Ok(false));
        assert_eq!(drain(&mut lifecycle, 4), Ok(false));
        assert!(!drained(&mut lifecycle, &mut false));
    }

    #[test]
    fn a_grow_applies_at_once_to_an_engine_that_serves() {
        let mut kept = false;
        assert!(grow(Lifecycle::Serving, &mut kept));
        assert!(!kept);
        // A stopped or failed engine grows nothing, now or later.
        for state in [Lifecycle::Stopped, Lifecycle::Failed] {
            let mut kept = false;
            assert!(!grow(state, &mut kept));
            assert!(!kept);
        }
    }

    #[test]
    fn a_grow_during_the_start_or_while_parked_is_kept_until_the_engine_serves() {
        // The start's session and regions do not exist yet, and a parked engine's regions keep
        // their sizes until its promotion.
        for state in [Lifecycle::Starting, Lifecycle::Parked] {
            let mut kept = false;
            assert!(!grow(state, &mut kept));
            assert!(kept, "{state:?}");
        }
    }

    #[test]
    fn a_promotion_that_overtakes_a_drain_grows_once_the_engine_serves_again() {
        let mut lifecycle = Lifecycle::Serving;
        let mut kept = false;
        assert_eq!(drain(&mut lifecycle, 4), Ok(true));

        // The promotion's grow, and a grow the other epoch's release sends meanwhile, wait for
        // the drain: its regions still shrink.
        assert_eq!(promote(&mut lifecycle, serving(LAUNCH, 1)), Ok(false));
        assert!(!grow(lifecycle, &mut kept));
        assert!(kept);

        // Drained and promoted, the engine serves again, and the kept grow applies.
        assert!(drained(&mut lifecycle, &mut kept));
        assert!(kept);
    }

    #[test]
    fn a_grow_kept_through_a_drain_that_parks_the_engine_is_dropped() {
        let mut lifecycle = Lifecycle::Serving;
        let mut kept = false;
        assert_eq!(drain(&mut lifecycle, 4), Ok(true));
        assert!(!grow(lifecycle, &mut kept));

        // No promotion: the memory the drain gave back is the serving epoch's.
        assert!(!drained(&mut lifecycle, &mut kept));
        assert!(!kept);
    }

    #[test]
    fn an_engine_that_is_not_ready_refuses_drain_and_promote() {
        for state in [Lifecycle::Starting, Lifecycle::Stopped, Lifecycle::Failed] {
            let mut lifecycle = state;
            let report = drain(&mut lifecycle, 4).expect_err("a drain was taken");
            assert!(matches!(kind(&report), report::Kind::Fault(_)));
            let report = promote(&mut lifecycle, serving(LAUNCH, 1)).expect_err("promoted");
            assert!(matches!(kind(&report), report::Kind::Fault(_)));
            assert_eq!(lifecycle, state);
        }
    }

    #[test]
    fn a_generation_is_prepared_switched_to_and_committed() {
        let mut update = Update::Steady;
        assert_eq!(prepare(Lifecycle::Serving, &mut update, 1), Ok(2));
        assert_eq!(update, Update::Preparing(2));

        assert!(prepared(&mut update, 2));
        assert_eq!(update, Update::Prepared(2));

        assert_eq!(switch(Lifecycle::Serving, &mut update, 2), Ok(()));
        assert_eq!(update, Update::Trial(2));

        assert_eq!(commit(&mut update, 2), Ok(2));
        assert_eq!(update, Update::Steady);
    }

    #[test]
    fn a_prepare_is_refused_unless_a_serving_engine_is_steady() {
        let mut update = Update::Steady;
        for lifecycle in [
            Lifecycle::Starting,
            Lifecycle::Parked,
            Lifecycle::Draining {
                serves_again: false,
            },
            Lifecycle::Stopped,
            Lifecycle::Failed,
        ] {
            let report = prepare(lifecycle, &mut update, 1).expect_err("prepared");
            assert!(matches!(kind(&report), report::Kind::PrepareFailed(_)));
            assert_eq!(update, Update::Steady);
        }

        for mut busy in [Update::Preparing(2), Update::Prepared(2), Update::Trial(2)] {
            let before = busy;
            let report = prepare(Lifecycle::Serving, &mut busy, 1).expect_err("a second update");
            assert!(matches!(kind(&report), report::Kind::PrepareFailed(_)));
            assert_eq!(busy, before);
        }
    }

    #[test]
    fn a_failed_preparation_ends_the_update_and_changes_nothing_else() {
        let mut update = Update::Steady;
        assert_eq!(prepare(Lifecycle::Serving, &mut update, 1), Ok(2));
        prepare_ended(&mut update, 2);
        assert_eq!(update, Update::Steady);

        // The switch the daemon may still send finds nothing prepared.
        let report = switch(Lifecycle::Serving, &mut update, 2).expect_err("switched");
        assert!(matches!(kind(&report), report::Kind::PrepareFailed(_)));
    }

    #[test]
    fn a_preparation_discarded_meanwhile_is_not_prepared() {
        let mut update = Update::Steady;
        assert_eq!(prepare(Lifecycle::Serving, &mut update, 1), Ok(2));
        assert_eq!(
            rollback(Lifecycle::Serving, &mut update, "abandoned"),
            Ok((2, false))
        );

        assert!(!prepared(&mut update, 2));
        assert_eq!(update, Update::Steady);
        // An end of the discarded preparation changes nothing either.
        prepare_ended(&mut update, 2);
        assert_eq!(update, Update::Steady);
    }

    #[test]
    fn a_switch_names_the_generation_a_serving_engine_prepared() {
        let mut update = Update::Steady;
        assert_eq!(prepare(Lifecycle::Serving, &mut update, 1), Ok(2));
        // Still preparing.
        assert!(switch(Lifecycle::Serving, &mut update, 2).is_err());
        assert!(prepared(&mut update, 2));
        // Another generation than the one prepared.
        assert!(switch(Lifecycle::Serving, &mut update, 3).is_err());
        // A drained engine reaches no batch boundary.
        let draining = Lifecycle::Draining {
            serves_again: false,
        };
        assert!(switch(draining, &mut update, 2).is_err());
        assert_eq!(update, Update::Prepared(2));

        assert_eq!(switch(Lifecycle::Serving, &mut update, 2), Ok(()));
        // A second switch finds the generation on trial already.
        let report = switch(Lifecycle::Serving, &mut update, 2).expect_err("switched twice");
        assert!(matches!(kind(&report), report::Kind::PrepareFailed(_)));
        assert_eq!(update, Update::Trial(2));
    }

    #[test]
    fn a_failed_switch_ends_the_trial() {
        let mut update = on_trial();
        assert!(switch_failed(&mut update, 2));
        assert_eq!(update, Update::Steady);
        assert!(!switch_failed(&mut update, 2));
    }

    #[test]
    fn a_commit_with_nothing_on_trial_is_answered_at_once() {
        let mut update = Update::Steady;
        let report = commit(&mut update, 1).expect_err("committed nothing");
        assert!(matches!(kind(&report), report::Kind::Committed(_)));

        for mut waiting in [Update::Preparing(2), Update::Prepared(2)] {
            let before = waiting;
            let report = commit(&mut waiting, 1).expect_err("committed what never executed");
            assert!(matches!(kind(&report), report::Kind::RolledBack(_)));
            assert_eq!(waiting, before);
        }
    }

    #[test]
    fn a_rollback_on_trial_switches_back_and_one_before_the_switch_discards() {
        let mut update = on_trial();
        assert_eq!(
            rollback(Lifecycle::Serving, &mut update, "the operator decided"),
            Ok((2, true))
        );
        assert_eq!(update, Update::Steady);

        for mut before in [Update::Preparing(2), Update::Prepared(2)] {
            assert_eq!(
                rollback(Lifecycle::Serving, &mut before, "abandoned"),
                Ok((2, false))
            );
            assert_eq!(before, Update::Steady);
        }
    }

    #[test]
    fn a_rollback_with_no_update_in_flight_is_acknowledged_with_its_reason() {
        let mut update = Update::Steady;
        let report = rollback(Lifecycle::Serving, &mut update, "a peer failed")
            .expect_err("rolled back nothing");
        assert_eq!(
            kind(&report),
            &report::Kind::RolledBack(proto::process::RolledBack {
                reason: "a peer failed".to_owned(),
            })
        );
        assert_eq!(update, Update::Steady);
    }

    #[test]
    fn a_fault_rolls_a_trial_back_and_changes_nothing_outside_one() {
        let mut update = on_trial();
        assert_eq!(fault_on_trial(Lifecycle::Serving, &mut update), Some(2));
        assert_eq!(update, Update::Steady);

        for mut outside in [Update::Steady, Update::Preparing(2), Update::Prepared(2)] {
            let before = outside;
            assert_eq!(fault_on_trial(Lifecycle::Serving, &mut outside), None);
            assert_eq!(outside, before);
        }
    }

    #[test]
    fn a_draining_engine_keeps_its_trial_until_it_is_drained() {
        let draining = Lifecycle::Draining {
            serves_again: false,
        };
        assert!(draining.boundary_refusal().is_some());
        for state in [Lifecycle::Serving, Lifecycle::Parked] {
            assert_eq!(state.boundary_refusal(), None, "{state:?}");
        }

        // A switch back would resume the schedulers mid-drain: the daemon's rollback is refused
        // and a fault rolls nothing back, the trial left as it is.
        let mut update = on_trial();
        let report = rollback(draining, &mut update, "the operator decided")
            .expect_err("switched back while draining");
        assert!(matches!(kind(&report), report::Kind::Fault(_)));
        assert_eq!(fault_on_trial(draining, &mut update), None);
        assert_eq!(update, Update::Trial(2));

        // What only a preparation built is discarded without a pause.
        let mut prepared = Update::Prepared(2);
        assert_eq!(
            rollback(draining, &mut prepared, "abandoned"),
            Ok((2, false))
        );

        // Once drained, the trial switches back.
        let parked = Lifecycle::Parked;
        assert_eq!(
            rollback(parked, &mut update, "the operator decided"),
            Ok((2, true))
        );
    }

    #[test]
    fn a_drain_discards_an_update_no_switch_took_and_keeps_a_trial() {
        for mut waiting in [Update::Preparing(2), Update::Prepared(2)] {
            assert_eq!(discarded_by_drain(&mut waiting), Some(2));
            assert_eq!(waiting, Update::Steady);
        }

        let mut update = on_trial();
        assert_eq!(discarded_by_drain(&mut update), None);
        assert_eq!(update, Update::Trial(2));

        let mut steady = Update::Steady;
        assert_eq!(discarded_by_drain(&mut steady), None);
    }
}
