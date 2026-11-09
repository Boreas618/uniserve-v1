//! The epoch layer's commands to this process: a drain that parks every scheduler at its boundary,
//! the `Serve` that resumes it, a standby's promotion, and admission, which opens only once the
//! epoch's first switchover and its warmup have run.

use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use ipc::UpdateFailure;

use super::MainProcess;
use super::device_command;

/// A drain whose schedulers have not all parked yet.
pub(super) struct Draining {
    /// The epoch `Drained` names.
    epoch: u64,
    /// The schedulers still to answer, in order.
    answers: Vec<crossbeam_channel::Receiver<std::result::Result<scheduler::ParkedCache, String>>>,
    /// What the ones that answered parked with, in the schedulers' order.
    parked: Vec<scheduler::ParkedCache>,
    /// When the drain began, which the stall bound is measured from.
    since: Instant,
}

/// Why a promotion this process cannot serve is refused, or `None` when it can serve.
///
/// A standby is promoted to serve the generation the epoch prepared in it, and nothing in
/// `Promote` states that generation - the message is empty and `Promoted` answers with the launch
/// number the epoch layer promotes by - so this process cannot tell a generation the epoch still
/// holds from one it has reverted. What it *can* tell is whether it holds a generation at all: a
/// promotion with nothing prepared is a process that would open admission with no kernel and no
/// graph, and answer `Promoted` to a daemon that then moves the frontend onto it.
fn promote_refusal(standby: bool, prepared: usize) -> Option<String> {
    (standby && prepared == 0).then(|| {
        "this standby holds no prepared generation: its precopy was discarded or never ran, so          promoting it would serve a generation this process does not hold"
            .to_owned()
    })
}

impl MainProcess {
    /// The epoch drains at the boundary its schedulers reach, and the process
    /// goes on living.
    ///
    /// Every scheduler parks: its migratable requests end with their facts kept, the others finish,
    /// and it executes nothing more. What it parks with is kept, so the `Serve` that names the epoch
    /// again resumes it in place instead of reading its pages back. Weights and storage stay mapped
    /// and the threads stay up - a drain is not an exit, and the daemon's `Serve` is what ends it.
    pub(super) fn drain(&mut self, epoch: u64) -> Result<()> {
        // An update cannot cross the drain: its switchover takes effect at a batch boundary and a
        // drained epoch reaches none, so what the precopy built is given back now rather than left
        // prepared for a boundary that cannot come.
        self.drop_prepared("the epoch drained before its switchover")?;
        // The daemon's client reads this before the first request ends migratable, on the same
        // connection, so a request that moves never finds this epoch ready: it waits for the epoch
        // the frontend serves next rather than being submitted to a process that admits nothing.
        self.announce(proto::request::wire::EngineState::Parked)?;
        // A scheduler parks once its requests have ended, and a request ends only when the daemon
        // acknowledges it through this thread, so the answers are polled from the serve pass:
        // waiting here would hold the acknowledgement the park is waiting for.
        self.draining = Some(Draining {
            epoch,
            answers: self
                .schedulers
                .iter()
                .map(|scheduler| scheduler.park(false))
                .collect(),
            parked: Vec::with_capacity(self.schedulers.len()),
            since: Instant::now(),
        });
        self.advance_drain()
    }

    /// Takes the answers of a drain in progress, and states `Drained` when every scheduler parked.
    pub(super) fn advance_drain(&mut self) -> Result<()> {
        let bound = self.watchdog.stall_bound();
        let Some(mut draining) = self.draining.take() else {
            return Ok(());
        };
        while let Some(answer) = draining.answers.first() {
            match answer.try_recv() {
                Ok(Ok(cache)) => {
                    draining.parked.push(cache);
                    draining.answers.remove(0);
                }
                Ok(Err(reason)) => bail!("draining the epoch: a scheduler did not park: {reason}"),
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    if draining.since.elapsed() > bound {
                        bail!("draining the epoch: a scheduler did not park within {bound:?}");
                    }
                    self.draining = Some(draining);
                    return Ok(());
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    bail!("draining the epoch: a scheduler is gone");
                }
            }
        }
        self.parked = Some(draining.parked);
        self.serving = false;
        // A drain ends the epoch this process was serving, so admission owed to a warmup of it is
        // owed to nothing: the `Serve` that names the epoch again opens the next one.
        self.admission_owed = false;
        self.send_report(proto::process::Report {
            kind: Some(proto::process::report::Kind::Drained(
                proto::process::Drained {
                    epoch: draining.epoch,
                },
            )),
        });
        // A revert that overtook the drain serves this epoch again, now that it can.
        if let Some(serve) = self.serve_after_drain.take() {
            self.resume(serve)?;
        }
        Ok(())
    }

    /// Gives back what an in-flight update prepared, by name.
    ///
    /// A precopy still running is abandoned through its own path, which discards the captures on
    /// every device, reinstates the serving policy and answers the peer or the daemon that asked.
    /// A generation already prepared is discarded the same way: the device queue orders the discard
    /// after the walk that captured it, and the generation that serves keeps its plans.
    fn drop_prepared(&mut self, reason: &str) -> Result<()> {
        if let Some(precopy) = self.precopy.take() {
            let generation = precopy.generation;
            let reply = precopy.reply;
            self.precopy = Some(precopy);
            return self.abandon_precopy(
                generation,
                UpdateFailure::Precopy {
                    reason: reason.to_owned(),
                },
                reply,
            );
        }
        let Some(generation) = self.prepared.first().map(|built| built.number()) else {
            return Ok(());
        };
        self.discard_policy();
        for worker in &self.workers {
            match worker.discard(generation) {
                Ok(submission) => {
                    if let Err(error) =
                        device_command(&self.watchdog, worker.device(), "discard", submission)
                    {
                        // The generation never serves: a discard that did not finish holds storage
                        // until the purge, and the epoch that drained is not made to fail for it.
                        tracing::warn!(%error, generation, "discarding a prepared generation failed");
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, generation, "queueing a prepared generation's discard failed");
                }
            }
        }
        self.prepared.clear();
        self.prepared_programs.clear();
        Ok(())
    }

    /// Epoch `serve.epoch` serves from now on, at this process.
    ///
    /// A process that drained resumes the schedulers it parked, with the cache it parked with; one
    /// that never drained already serves and has nothing to resume. The request address in the
    /// command is the one this process bound at launch: the daemon names it so that the frontend
    /// submits to the right place, not so that a process moves its socket.
    pub(super) fn resume(&mut self, serve: proto::process::Serve) -> Result<()> {
        if !self.boundary {
            // The daemon has said the epoch serves; the process cannot admit until its first
            // generation is filled and warmed. The request waits in the frontend, which retries the
            // submission it was refused.
            self.pending_serve = true;
            return Ok(());
        }
        if self.draining.is_some() {
            self.serve_after_drain = Some(serve);
            return Ok(());
        }
        let drained = self.parked.is_some();
        // A `Serve` that arrives while the warmup still has to capture this generation's buckets is
        // owed rather than refused: admission opens when the last submission finishes, the same way
        // it does at a switchover.
        self.open_admission()
            .with_context(|| format!("resuming epoch {}", serve.epoch))?;
        if drained {
            // A target ran on this cache's storage while this epoch stood drained, so what its
            // index names holds the target's pages now. The frontend moves only after the report
            // below, so every request comes after this command.
            for scheduler in &self.schedulers {
                scheduler.flush_cache();
            }
            // The daemon returns a drained epoch to serving and moves the frontend once
            // every host says so, as a promotion's answer says it of a target.
            self.send_report(proto::process::Report {
                kind: Some(proto::process::report::Kind::Promoted(
                    proto::process::Promoted {
                        generation: self.control.engine(),
                        program_generation: self.served_generation(),
                    },
                )),
            });
            self.announce(proto::request::wire::EngineState::Ready)?;
        }
        Ok(())
    }

    /// Promotes this standby in place.
    pub(super) fn promote_in_place(&mut self) -> Result<()> {
        // A standby is promoted to serve the generation the epoch prepared in it.
        // A promotion that finds nothing prepared - the precopy was discarded, or the
        // epoch moved on before it ran - is a promotion of a process that cannot serve,
        // and it is refused here, by name, before admission opens: `Promoted` followed
        // by a first request that finds no graph would tell the daemon the promotion
        // succeeded and move the frontend onto a process with no kernels.
        // A standby that took the epoch's programs serves them already: its precopy, switchover
        // and commit ran when the daemon handed them over, so its generation is the serving one.
        let held = self.prepared.len()
            + self
                .generations
                .iter()
                .filter(|generation| !generation.is_empty())
                .count();
        if let Some(reason) = promote_refusal(self.standby, held) {
            tracing::warn!(%reason, "a promotion this process cannot serve is refused");
            self.send_report(proto::process::Report {
                kind: Some(proto::process::report::Kind::Fault(proto::process::Fault {
                    kind: "promote".to_owned(),
                    reason: reason.clone(),
                    device: 0,
                })),
            });
            bail!("{reason}");
        }
        self.promoted = true;
        // The generation the schedulers must stamp this process's batches with: the one it
        // holds. A promotion is the one cut-over that reaches no `switchover` of this
        // process's own, so nothing else records it, and a promoted standby whose batches
        // were stamped 0 would find no graph - the same defect the switchover had.
        self.record_generation(self.served_generation());
        // A promoted standby admits on the same rule as any other process: once the
        // generation it will serve has been warmed.
        self.open_admission()
            .context("promoting this process in place")?;
        if !self.boundary {
            // A promoted standby serves only once the switchover that fills its
            // generation has happened, the same rule a fresh process follows.
            self.serving = false;
            self.pending_serve = true;
        }
        // The two numbers answer different questions and both are needed: the launch
        // generation is what the epoch layer promotes a child by, and the program
        // generation is the one this process holds - the fact the daemon compares with
        // its own before it moves the frontend, so a promotion of a generation the
        // epoch has reverted is refused there instead of served here.
        self.send_report(proto::process::Report {
            kind: Some(proto::process::report::Kind::Promoted(
                proto::process::Promoted {
                    generation: self.control.engine(),
                    program_generation: self.served_generation(),
                },
            )),
        });
        // The request link says so too: the daemon's client waits on it to move its frontend.
        self.announce(proto::request::wire::EngineState::Ready)?;
        Ok(())
    }

    /// Opens admission, or owes it to the warmup.
    ///
    /// A bucket is served only after its capture has run: a replay reads the captures of the
    /// generation serving, and a bucket captured by no one is an invariant error at the first batch
    /// rather than a capture that is slow. `warmup` queues one capture of every bucket per device
    /// and returns at once - it runs one bucket per queue turn behind serving, which for a model
    /// with many buckets is minutes - so the switchover answers the daemon and leaves the opening
    /// here. A `Drain` clears what was owed, because the epoch it was owed to is no longer serving.
    pub(super) fn open_admission(&mut self) -> Result<()> {
        if !self.warming.is_empty() {
            tracing::info!(
                warming = self.warming.len(),
                "admission is owed to the warmup, which has not finished"
            );
            self.admission_owed = true;
            return Ok(());
        }
        tracing::info!(
            boundary = self.boundary,
            pending_serve = self.pending_serve,
            "opening admission"
        );
        self.admission_owed = false;
        // An early `Serve` is honoured here rather than at its arrival: this is the pass that
        // admits, so the request waiting in the frontend is exactly what it was waiting for.
        self.pending_serve = false;
        // What the ranks hold to serve is warm by now, so the ledger hears of it before any request
        // can ask for more. A rank past its quota does not serve.
        self.data
            .book_serving()
            .context("booking what the ranks hold in their devices' ledgers")?;
        self.resume_schedulers()
    }

    /// Parks every scheduler, keeping what each parks with for the resume.
    ///
    /// Each park is bounded by the watchdog's stall bound, the same rule the scheduler stage calls
    /// follow: a scheduler that cannot park is a stall the epoch has to name rather than a
    /// control thread that waits forever.
    pub(super) fn park_schedulers(&mut self, what: &str) -> Result<()> {
        let bound = self.watchdog.stall_bound();
        let mut parked = Vec::with_capacity(self.schedulers.len());
        for scheduler in &self.schedulers {
            match scheduler.park(false).recv_timeout(bound) {
                Ok(Ok(cache)) => parked.push(cache),
                Ok(Err(reason)) => bail!("{what}: a scheduler did not park: {reason}"),
                Err(error) => bail!("{what}: a scheduler did not park within {bound:?}: {error}"),
            }
        }
        self.parked = Some(parked);
        self.serving = false;
        // A drain ends the epoch this process was serving, so admission owed to a warmup of it is
        // owed to nothing: the `Serve` that names the epoch again opens the next one.
        self.admission_owed = false;
        Ok(())
    }

    /// Resumes every scheduler with the cache it parked with: the pages it still holds are reused
    /// and the facts the drain kept are resubmitted by the frontend.
    pub(super) fn resume_schedulers(&mut self) -> Result<()> {
        let Some(parked) = self.parked.take() else {
            self.serving = true;
            return Ok(());
        };
        let bound = self.watchdog.stall_bound();
        for (scheduler, cache) in self.schedulers.iter().zip(parked) {
            // A resumed epoch recomputes from the tokens the frontend resubmits rather than lending
            // from a predecessor: it is the same epoch, and the requests it ended at the drain are
            // new requests to it.
            scheduler
                .promote(cache, true, false)
                .context("resuming a scheduler")?;
            // The promotion states the cache; it is the resume that ends the park. A scheduler left
            // parked takes requests into its queue and admits none of them.
            match scheduler.resume(true).recv_timeout(bound) {
                Ok(Ok(())) => {}
                Ok(Err(reason)) => bail!("resuming a scheduler: {reason}"),
                Err(error) => bail!("resuming a scheduler: no answer within {bound:?}: {error}"),
            }
        }
        self.serving = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A promotion with nothing prepared is refused, and one with a generation is not: the engine
    /// cannot see the epoch's current generation (`Promote` states none), so what it can refuse is
    /// a process that holds none at all.
    #[test]
    fn a_promotion_with_nothing_prepared_is_refused_and_one_with_a_generation_is_not() {
        assert!(
            promote_refusal(true, 0).is_some(),
            "a standby with nothing prepared would have been promoted into an empty generation"
        );
        assert!(
            promote_refusal(true, 1).is_none(),
            "a standby holding its prepared generation was refused"
        );
        assert!(
            promote_refusal(false, 0).is_none(),
            "an initial process, which is never promoted, was refused"
        );
        let reason = promote_refusal(true, 0).unwrap_or_default();
        assert!(reason.contains("no prepared generation"), "{reason}");
        assert!(reason.contains("discarded"), "{reason}");
    }
}
