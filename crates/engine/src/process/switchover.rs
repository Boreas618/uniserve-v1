//! The program-layer update after its precopy: the switchover at a batch boundary and the warmup
//! it queues, probation, the commit, the revert to the generation that served before, and the
//! purge of a generation after its last batch.

use std::time::Instant;

use anyhow::{Context as _, Result};
use ipc::{ProgramStatus, UpdateFailure};
use worker::completion::Outcome;

use super::MainProcess;
use super::device_command;

/// What a switchover answers: the status now, or nothing until its warmup has run.
///
/// `Switched` states that this process serves, and it cannot serve a batch before the buckets that
/// batch may be planned into exist - so a switchover that queued a warmup answers after admission
/// opens, and a warmup that fails answers the switchover with the failure instead. A switchover with
/// nothing to warm has nothing to wait for.
#[derive(Debug)]
enum SwitchoverAnswer {
    Now(ProgramStatus),
    AfterTheWarmup,
}

fn switchover_answer(warming: usize, generation: u64) -> SwitchoverAnswer {
    if warming == 0 {
        SwitchoverAnswer::Now(ProgramStatus::Switched { generation })
    } else {
        SwitchoverAnswer::AfterTheWarmup
    }
}

/// The answer to a `Commit`.
///
/// A commit whose generation is not on probation names nothing the machine holds: the daemon may be
/// one message behind the revert that its own rollback threshold caused, so its commit arrives after
/// the update ended. The state that holds is the one before the update - it serves, nothing was
/// committed - and that is the answer. A protocol message never ends a process that can still serve.
fn answer_commit(update: &mut worker::hot_update::HotUpdate, generation: u64) -> ProgramStatus {
    // An epoch switch commits its target, whose programs were committed when the daemon handed them
    // over: the generation it names already serves with nothing on trial, so the answer is that.
    if update.current() == generation
        && matches!(update.phase(), worker::hot_update::Phase::Serving)
    {
        return ProgramStatus::Committed { generation };
    }
    match update.commit(generation) {
        Ok(()) => ProgramStatus::Committed { generation },
        Err(error) => {
            tracing::warn!(
                generation,
                serving = update.current(),
                %error,
                "a commit names a generation that is not on probation"
            );
            ProgramStatus::Reverted {
                generation,
                reason: format!(
                    "the commit names a generation that is not in flight ({error}): generation {} \
                     serves",
                    update.current()
                ),
            }
        }
    }
}

impl MainProcess {
    /// Reads the warmups the switchovers queued: a finished one leaves the list, and one
    /// that failed reverts the generation it was warming.
    ///
    /// A warmup is the first run of every captured bucket of the generation that just took over, so
    /// it is where a capture that cannot be made on this device surfaces - after the daemon was told
    /// the cut-over happened, which is why the failure goes through the update's decision table
    /// rather than through the switchover's answer.
    pub(super) fn advance_warmup(&mut self) -> Result<Option<UpdateFailure>> {
        // A warmup still running past twice the stall bound is a device that will not finish it: its
        // answer is owed to the daemon's switchover wait, so silence here is a timeout with no cause
        // in the log. Failing it by name reverts the update and leaves the process serving what it
        // held.
        if let (Some(since), true) = (self.warming_since, !self.warming.is_empty())
            && since.elapsed() > self.watchdog.stall_bound() * 2
        {
            let (device, generation) = self
                .warming
                .first()
                .map_or((0, 0), |(device, _, generation)| (*device, *generation));
            let failure = UpdateFailure::Device {
                device,
                error: format!(
                    "warming generation {generation} on device {device} did not finish in {}s",
                    since.elapsed().as_secs()
                ),
                batch: None,
            };
            self.warming.clear();
            self.warming_since = None;
            self.pending_switch = None;
            return self.failure(failure);
        }
        let mut pending = Vec::with_capacity(self.warming.len());
        for (device, submission, generation) in std::mem::take(&mut self.warming) {
            match submission.outcome() {
                None => pending.push((device, submission, generation)),
                Some(Outcome::Success(_)) => {
                    tracing::info!(device, generation, "a generation's warmup finished");
                }
                Some(outcome) => {
                    self.warming = pending;
                    let reason = match outcome {
                        Outcome::Failed(error) => error.to_string(),
                        _ => "the warmup was cancelled".to_owned(),
                    };
                    let failure = UpdateFailure::Device {
                        device,
                        error: format!(
                            "warming generation {generation} on device {device}: {reason}"
                        ),
                        batch: None,
                    };
                    // The answer owed to the switchover goes out from the revert below, as
                    // `Reverted` - the status the daemon's wait accepts for a generation that took
                    // over and went back. Reading it as served is what this ordering prevents.
                    self.pending_switch = None;
                    self.warming_since = None;
                    // The table decides, as it does for every other failure: on probation this
                    // reverts the generation, and an update the daemon already committed leaves a
                    // device that cannot run it, which ends the epoch rather than serving on.
                    return self.failure(failure);
                }
            }
        }
        self.warming = pending;
        if self.warming.is_empty() && self.admission_owed {
            // The last capture of the epoch's warmup has run, so every bucket a batch can be planned
            // into exists. A warmup that failed returns above and its revert clears what was owed.
            self.open_admission()
                .context("opening admission after the warmup")?;
        }
        // The switchover's answer is owed to the warmup, and it is sent here - after admission
        // opened - so that `Switched` states the one thing it means: this process can serve.
        if self.warming.is_empty()
            && let Some((generation, reply)) = self.pending_switch.take()
        {
            tracing::info!(
                generation,
                "answering the switchover now that admission is open"
            );
            self.answer(reply, ProgramStatus::Switched { generation });
        }
        Ok(None)
    }

    /// The switchover: cut over at the batch boundary whose step the scheduler put
    /// the command at; an expert group's ranks switch at the same step.
    pub(super) fn switchover(
        &mut self,
        generation: u64,
        step: Option<u64>,
        reply: Option<u32>,
    ) -> Result<Option<ProgramStatus>> {
        // A switchover to a generation this process never prepared cannot succeed, and cutting the
        // devices over to try it costs a live graph rebuild to end where it started: it is refused
        // before any device is touched, and the generation that serves keeps serving.
        //
        // "Prepared nothing" and "prepared another" are the same refusal to the daemon and two
        // different bugs in the log, so the reason names which one. The second is the state a lost
        // `Precopied` reply leaves: the daemon gave up on a precopy that then completed, retried
        // `/admin/config`, and now names a generation this process never built. Letting it through
        // would move the scheduling policy to `generation` first, then fail on the devices, and
        // report a device fault for a stale number.
        let stale = self
            .prepared
            .iter()
            .map(|built| built.number())
            .find(|number| *number != generation);
        let refused = if self.prepared.is_empty() {
            Some(format!(
                "a switchover to generation {generation}, which this process prepared nothing \
                 for: the precopy that builds it either failed or never arrived"
            ))
        } else {
            stale.map(|other| {
                format!(
                    "a switchover to generation {generation}, while this process holds generation \
                     {other} prepared and never cut over: the precopy that builds {generation} \
                     either failed or never arrived"
                )
            })
        };
        if let Some(reason) = refused {
            return Ok(Some(ProgramStatus::PrecopyFailed {
                generation,
                failure: UpdateFailure::Precopy { reason },
            }));
        }
        // Before the devices cut over and before admission can open on a batch: the generation a
        // step stamps its batches with is the one whose graphs the runner has to find.
        self.record_generation(generation);
        self.switchover_policy(generation, step);
        for worker in &self.workers {
            let device = worker.device();
            let submission = worker.switchover(generation)?;
            if let Err(error) = device_command(&self.watchdog, device, "switchover", submission) {
                // A device that did not cut over would leave this node serving two generations at
                // once, so the revert runs on every device: the generation that served
                // before the update is the one serving after it.
                tracing::error!(%error, "cutting over failed; reverting the update");
                let failure = UpdateFailure::Device {
                    device: u32::try_from(device.0).unwrap_or_default(),
                    error: error.to_string(),
                    batch: None,
                };
                return self.revert(generation, failure).map(Some);
            }
        }
        // The record moves before the warmup: probation begins the moment every device serves the
        // new generation, and the warmup is the last part of the cut-over rather than of the
        // generation that served before it. So a warmup that fails is a revert of a generation that
        // *served* - a reverse switchover on every device - and not a discard of one that never
        // cut over, which is what the phase decides, and it cannot be asked before this line.
        if let Err(error) = self.update.switchover(generation, Instant::now()) {
            // The devices cut over and the machine did not: a switchover for a generation the
            // update is not on probation for is a stale message, and the state that holds is the
            // generation that served. The reverse switchover runs, and its answer is what the
            // daemon reads - a protocol message does not end a process that can still serve.
            tracing::warn!(generation, %error, "a switchover names no generation on probation");
            let failure = UpdateFailure::Threshold {
                reason: format!(
                    "the switchover names a generation that is not on probation: {error}"
                ),
            };
            return self.revert(generation, failure).map(Some);
        }
        self.retired = std::mem::replace(&mut self.generations, std::mem::take(&mut self.prepared));
        // The programs move with the generations, as one record: the next precopy's changed set is
        // measured against what serves, and a revert serves the programs the retired pair holds.
        self.retired_programs = std::mem::replace(
            &mut self.serving_programs,
            std::mem::take(&mut self.prepared_programs),
        );

        // The first request served on a generation must not pay for a capture, so every
        // captured bucket of the generation that just took over runs once, here, between batches.
        // A switchover later in the epoch warms the buckets the new kernel programs changed; the
        // first one also warms the buckets no update touched, because until now nothing served
        // them.
        // The warmups are queued, not waited for: each is a device job that runs one bucket per
        // queue turn behind serving batches, so a model with many buckets warms for minutes while
        // the daemon waits two minutes for `Switched` - and a control thread that waited here would
        // answer no command, no request event and no metrics for the same minutes. `advance_warmup`
        // reads them on the passes that follow, and a device that fails reverts the update.
        for worker in &self.workers {
            let device = worker.device();
            let ordinal = u32::try_from(device.0).unwrap_or_default();
            match worker.warmup() {
                Ok(submission) => self.warming.push((ordinal, submission, generation)),
                Err(error) => {
                    let failure = UpdateFailure::Device {
                        device: ordinal,
                        error: format!("queueing the warmup of generation {generation}: {error}"),
                        batch: None,
                    };
                    if let Some(ending) = self.failure(failure)? {
                        return Ok(Some(ProgramStatus::Failed {
                            generation,
                            failure: ending,
                        }));
                    }
                    return Ok(Some(ProgramStatus::Switched { generation }));
                }
            }
        }

        tracing::info!(
            generation,
            devices = self.workers.len(),
            queued = self.warming.len(),
            "the switchover queued the epoch's warmup and set the boundary"
        );
        // The boundary the first batch is served at: every device serves this generation
        // and its buckets are warming behind the batches that follow. A `Serve` that arrived while
        // the generation was still being filled is honoured now.
        self.boundary = true;
        // An epoch admits once this boundary is behind it and nothing has parked it since: a
        // non-standby process was launched to serve, a standby only after the daemon promoted it,
        // and a drained one waits for the `Serve` that names the epoch again. The start of an epoch
        // carries no `Serve` - the daemon tells the frontend where requests go and the process was
        // launched serving - so the boundary itself is what opens admission.
        let admitted = self.pending_serve || self.parked.is_none();
        if admitted && (self.promoted || !self.standby) {
            // Admission is the end of the warmup, not the boundary: the buckets the epoch serves
            // are the ones the warmup captures, and a batch admitted before the last of them has
            // run and been captured would be planned into a bucket with no graph. So the answer to
            // the daemon leaves `Switched` inside its window and the opening happens in
            // `advance_warmup`, which is what honours an early `Serve` when a drain parked the
            // schedulers first - a process that never parked resumes nothing and simply admits.
            self.open_admission()
                .context("resuming for the epoch's first switchover")?;
        }
        // The other silence of the switchover: a first switchover that installed no policy adopts
        // nothing and says nothing, so this process asks the policy slots once and, if none serves,
        // asks again one stall bound later before it names the failure. The check is queued here
        // rather than run here, so the daemon's `Switched` is never delayed by it.
        self.switchovers = self.switchovers.saturating_add(1);
        if self.switchovers == 1 {
            self.policy_probe_at = Some(Instant::now());
        }
        match switchover_answer(self.warming.len(), generation) {
            SwitchoverAnswer::Now(status) => Ok(Some(status)),
            // `Switched` states that this process serves, and it cannot serve until the buckets a
            // batch may be planned into exist - admission opens in `advance_warmup`. So the answer
            // waits for it: a warmup that fails then answers with the failure, and the daemon never
            // opens a frontend onto a process that cannot take a request.
            SwitchoverAnswer::AfterTheWarmup => {
                self.pending_switch = Some((generation, reply));
                self.warming_since = Some(Instant::now());
                Ok(None)
            }
        }
    }

    /// The commit: probation ended in a commit; the previous generation is kept
    /// until its purge.
    pub(super) fn commit(&mut self, generation: u64) -> Result<ProgramStatus> {
        let status = answer_commit(&mut self.update, generation);
        if matches!(status, ProgramStatus::Committed { .. }) {
            // Probation ended in a commit, so the generation before the update can never serve
            // again. Its graphs and kernels live until the purge, which is the worker's book to
            // keep - and the programs it was built from go with the record of it, since no revert
            // can serve it.
            self.retired.clear();
            self.retired_programs.clear();
            self.commit_policy(generation);
        }
        Ok(status)
    }

    /// Probation: an update left on probation past the window the launch file states is
    /// reverted, so a deployment never serves a generation nobody accepted.
    ///
    /// The daemon commits; this is the guard for when it does not - an operator who closed the
    /// dashboard, a commit that never arrived. Without a stated window there is no guard, which is
    /// the daemon's decision and not this process's.
    pub(super) fn watch_probation(&mut self) -> Result<Option<UpdateFailure>> {
        let Some(window) = self.rollback else {
            return Ok(None);
        };
        if !self.update.probation_expired(Instant::now(), window) {
            return Ok(None);
        }
        let failure = UpdateFailure::Threshold {
            reason: format!(
                "the update was on probation for longer than the {} seconds the launch file allows",
                window.as_secs()
            ),
        };
        tracing::info!(
            seconds = window.as_secs(),
            "an update outlived its probation and is reverted"
        );
        // A device that cannot put the generation that served back is a device with no generation at
        // all: the revert's refusal is the epoch's, and it ends by that name rather than continuing to
        // serve a generation the process has just decided it cannot stand behind.
        let generation = self.update.current();
        self.revert(generation, failure)?;
        Ok(None)
    }

    /// The revert: back to the generation serving before the update. The old
    /// generation and its nodes stayed through probation, so the revert is the
    /// reverse switchover and nothing has to be prepared again.
    pub(super) fn revert(
        &mut self,
        generation: u64,
        failure: UpdateFailure,
    ) -> Result<ProgramStatus> {
        self.policy_probe_at = None;
        self.policy_missing_since = None;
        // The phase before the call decides what a device does with the update's generation: a
        // generation that was prepared but never cut over never served, so nothing goes back and it
        // is discarded; a generation on probation served, and its reverse switchover runs. The
        // machine answers `Serving` either way once the call returns, so the phase is read first.
        let phase = self.update.phase();
        if let Err(error) = self.update.revert(generation, failure.clone()) {
            tracing::error!(%error, "reverting a program-layer update failed");
            return Ok(ProgramStatus::Failed {
                generation,
                failure,
            });
        }
        // The batches that form from here serve the generation that served before the update, so the
        // number they are stamped with is that one: a revert that left the update's number in place
        // would stamp every batch with a generation whose graphs are about to be released.
        self.record_generation(self.update.current());
        self.revert_policy(generation);
        // The generation that served before the update serves again, and the one built for the
        // update keeps nothing: its kernels are released and its captures are never taken over.
        let kept = !self.retired.is_empty();
        if kept {
            self.generations = std::mem::take(&mut self.retired);
            self.serving_programs = std::mem::take(&mut self.retired_programs);
        }
        self.prepared.clear();
        self.prepared_programs.clear();
        // The warmups still queued warm the generation this revert went back from: what runs when
        // they reach the device is the generation that serves, so nothing hangs - the list is what
        // stopped describing serving, and a failure read from a stale entry would revert an update
        // that is over.
        self.warming.clear();
        self.warming_since = None;
        // The answer owed to that warmup goes with it: the daemon's switchover wait is answered by
        // the revert's own report, and a later `Switched` would state a generation that does not
        // serve.
        self.pending_switch = None;
        // What the update's warmup still owed is owed no longer. An epoch that kept the generation
        // which served before it goes on serving, and that generation's buckets are captured - a
        // cancelled warmup of the update's is nothing to wait for. A revert of the epoch's first
        // switchover keeps nothing, and its end is the report rather than an admission.
        let owed = std::mem::take(&mut self.admission_owed);
        if owed && kept && self.boundary && (self.promoted || !self.standby) {
            self.resume_schedulers()
                .context("resuming the generation a revert went back to")?;
        }
        let current = self.update.current();
        for worker in &self.workers {
            let device = worker.device();
            // The revert is the reverse switchover, and `Worker::revert` is the verb
            // that runs it: a `switchover` back to the kept generation would find nothing prepared
            // (the cut-over took `next`) and refuse by name - the current generation would be left
            // serving with no kernels and no graphs.
            // A device that cannot go back to the generation that served is a device with no
            // generation at all: the update's kernels are the ones its graphs were rebound to, and a
            // revert that half happened leaves the epoch serving from neither. So a failed revert
            // ends the epoch by name instead of leaving it to fail on a batch that finds no graph.
            let queued = match phase {
                // The precopy's discard: the generation's graphs and kernels go, and the generation
                // that serves is untouched. A reverse switchover here would be refused by name -
                // nothing was switched over - and would end the epoch over an update that never cut.
                worker::hot_update::Phase::Precopied { .. } => worker.discard(generation),
                _ => worker.revert(current),
            };
            let submission = queued.with_context(|| {
                format!(
                    "queueing device {}'s revert to generation {current}",
                    device.0
                )
            })?;
            device_command(&self.watchdog, device, "revert", submission)?;
        }
        // No purge follows a revert: the daemon purges the generation a *commit* released, and its
        // reverted path sends nothing. The generation that served and went back
        // would otherwise keep its graphs, its kernels and its arena until the process exits - a
        // whole generation of device memory - so it is released here, now that every device has
        // finished going back and no batch of it can be in flight.
        if matches!(phase, worker::hot_update::Phase::Probation { .. }) {
            match self.purge(generation) {
                Ok(ProgramStatus::Purged { .. }) => {}
                Ok(other) => tracing::warn!(
                    ?other,
                    "the generation the revert went back from was not released"
                ),
                Err(error) => {
                    // The revert stands: serving is correct and what a failed release costs is the
                    // reverted generation's storage, which the process keeps until it exits.
                    tracing::error!(%error, "releasing the generation a revert went back from failed");
                }
            }
        }
        Ok(ProgramStatus::Reverted {
            generation,
            reason: failure.reason(),
        })
    }

    /// The purge: release the generation before the serving one, once its last
    /// batch completed. The worker's queue orders the purge behind every batch
    /// it already holds, and `Captured` keeps each kernel alive until the graph
    /// that uses it is rebound or destroyed.
    pub(super) fn purge(&mut self, generation: u64) -> Result<ProgramStatus> {
        // A purge releases what this process holds of a generation, so a number it has already
        // released is answered rather than refused: the revert below releases a reverted generation
        // itself, and a daemon that later purges the same number asks for a release that happened.
        if self.released.contains(&generation) {
            return Ok(ProgramStatus::Purged { generation });
        }
        if let Err(error) = self.update.request_purge(generation) {
            // A purge for a generation this process holds nothing of - one it never built, or one a
            // revert already released - asks for a release that has happened, and the honest answer
            // is that this process holds nothing of it. The warning is what tells a race from a
            // store-side mistake; neither is a reason to end a serving process.
            tracing::warn!(generation, %error, "a purge names a generation no update holds");
            self.released.insert(generation);
            return Ok(ProgramStatus::Purged { generation });
        }
        // A purge has no policy half: the scheduling policy is one program per epoch and its
        // modules are released with the scheduler, not with a generation's graphs.
        for worker in &self.workers {
            let device = worker.device();
            let submission = worker.purge(generation)?;
            if let Err(error) = device_command(&self.watchdog, device, "purge", submission) {
                // The generation still serves nothing, so a purge that did not finish leaks its
                // storage rather than breaking correctness: it is reported and the epoch serves on.
                let failure = UpdateFailure::Device {
                    device: u32::try_from(device.0).unwrap_or_default(),
                    error: error.to_string(),
                    batch: None,
                };
                return Ok(ProgramStatus::Failed {
                    generation,
                    failure,
                });
            }
        }
        if let Err(error) = self.update.purged(generation) {
            // The release happened on every device; only the bookkeeping did not recognise the
            // generation. Recording it is the whole of the answer, and the warning keeps the race
            // in the log rather than in the exit code.
            tracing::warn!(generation, %error, "a purged generation no update held");
        }
        self.released.insert(generation);
        Ok(ProgramStatus::Purged { generation })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The state a daemon's late messages meet: an update that went to probation and was reverted by
    /// the rollback threshold, so every command still in flight names a generation no longer there.
    fn update_on_probation_then_reverted() -> worker::hot_update::HotUpdate {
        let mut update = worker::hot_update::HotUpdate::new(1);
        update.begin_precopy(2).expect("a precopy may begin");
        update.precopied(2).expect("the precopy lands");
        update
            .switchover(2, Instant::now())
            .expect("the switchover takes effect");
        update
            .revert(
                2,
                UpdateFailure::Threshold {
                    reason: "the rollback threshold".into(),
                },
            )
            .expect("the revert lands");
        update
    }

    /// An epoch switch's commit reaches a target whose programs were committed at their handover:
    /// the generation it names serves with nothing on trial, so it is answered as committed: the
    /// switch waits for that acknowledgement before the previous epoch's process ends.
    #[test]
    fn a_commit_of_the_generation_that_serves_is_committed() {
        let mut update = update_on_probation_then_reverted();
        match answer_commit(&mut update, 1) {
            ProgramStatus::Committed { generation } => assert_eq!(generation, 1),
            other => panic!("a commit of what serves answered {other:?}"),
        }
    }

    /// A commit that races the revert is answered with the state that holds, and the process keeps
    /// serving the generation before the update: a message whose whole job is to be answered never
    /// ends its answerer.
    #[test]
    fn a_commit_after_a_revert_answers_with_the_generation_that_serves() {
        let mut update = update_on_probation_then_reverted();
        match answer_commit(&mut update, 2) {
            ProgramStatus::Reverted { generation, reason } => {
                assert_eq!(generation, 2);
                assert!(
                    reason.contains("not in flight"),
                    "the answer does not say why the commit did not apply: {reason}"
                );
                assert!(
                    reason.contains("generation 1 serves"),
                    "the answer does not name what serves instead: {reason}"
                );
            }
            other => panic!("a raced commit answered {other:?} rather than the state that holds"),
        }
        assert_eq!(
            update.current(),
            1,
            "the generation before the update does not serve after a raced commit"
        );
    }

    /// The rest of the family after the same revert. The reverted generation is released, so its
    /// purge is the one command that applies and a second purge waits behind it; a
    /// commit, a second revert and a second precopy each name a generation no longer in flight, and
    /// the machine answers each without leaving the generation that serves.
    #[test]
    fn the_program_commands_after_a_revert_all_name_what_they_cannot_apply() {
        let mut update = update_on_probation_then_reverted();
        update
            .request_purge(2)
            .expect("a revert releases its generation for the purge that frees its kernels");
        assert!(
            update.request_purge(2).is_err(),
            "a second purge of the generation applied while the first is pending"
        );
        assert!(update.commit(2).is_err());
        assert!(
            update
                .revert(
                    2,
                    UpdateFailure::Threshold {
                        reason: "again".into()
                    }
                )
                .is_err(),
            "a revert of a generation that is not on probation applied"
        );
        assert!(
            update.precopied(2).is_err(),
            "a precopy with none in flight applied"
        );
        assert_eq!(
            update.current(),
            1,
            "the serving generation moved under commands it could not apply"
        );
    }
}
