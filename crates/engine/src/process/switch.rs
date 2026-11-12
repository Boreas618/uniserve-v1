//! A generation taking over: the switch at a batch boundary and the captures it queues, the
//! commit, the rollback to the generation that executed before, and the release of the generation
//! an update leaves behind.
//!
//! An engine's start installs its first generation through the same switch
//! ([`MainProcess::install_first`]); the start then ends when the captures do, and a switch of a
//! serving engine is acknowledged then.

use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use ipc::UpdateFailure;
use ipc::execution::Outcome;
use ipc::execution::Submission;

use super::MainProcess;
use super::device_command;
use super::lifecycle::Lifecycle;

/// A switch whose generation executes on every device and whose captures still run.
///
/// The first request served on a generation must not pay for a capture, so every captured bucket
/// of the generation that took over runs once, between batches: one device job per device, one
/// bucket per queue turn behind serving. A model with many buckets captures for minutes, so the
/// jobs are queued and their completions wake the control thread.
pub(super) struct Switching {
    generation: u64,
    /// The capture job of each device that has not finished, by device ordinal.
    captures: Vec<(u32, Submission)>,
    /// The schedulers' responses to the policy switch that have not arrived: each responds at
    /// its next batch boundary.
    policies: Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>,
    /// The instant by which every scheduler must have responded.
    policy_deadline: Instant,
    /// When a device last moved on to another capture, which the captures' bound is measured
    /// from: a deployment's buckets decide how many captures there are, so the bound is on one
    /// that stops moving, never on their number.
    moved: Instant,
    /// The step each capturing device stood on when the thread last ran, in `captures` order.
    steps: Vec<Option<String>>,
}

impl Switching {
    /// The instant the switch is failed for standing still: a capture that did not move on
    /// within twice `stall`, or a scheduler that did not respond in time.
    pub(super) fn bound(&self, stall: Duration) -> Instant {
        let captures = self.moved + stall * 2;
        if self.policies.is_empty() {
            captures
        } else {
            captures.min(self.policy_deadline)
        }
    }
}

impl MainProcess {
    /// The start's install: the generation the start prepared executes as the engine's first.
    /// Nothing executed before it, so nothing is kept and nothing can be rolled back to: a device
    /// that cannot take it fails the start.
    pub(super) fn install_first(&mut self, generation: u64) -> Result<Option<UpdateFailure>> {
        if let Err(failure) = self.cut_over(generation) {
            return self.failure(failure);
        }
        self.state
            .generations
            .install_first(generation)
            .context("installing the engine's first generation")?;
        self.capture(generation)
    }

    /// The switch: every device and every scheduler executes `generation` from the next batch
    /// boundary, and the generation that executed until now is kept for the trial.
    pub(super) fn switch(&mut self, generation: u64) -> Result<Option<UpdateFailure>> {
        tracing::info!(generation, "switching to the prepared generation");
        if let Err(failure) = self.cut_over(generation) {
            // A device that did not switch would leave this node executing two generations at
            // once, so the generation is given back on every device: the one that executed
            // before the update is the one executing after it.
            tracing::error!(reason = %failure.reason(), "switching failed; the update is dropped");
            let (left, _) = self.state.rollback_on_failure()?;
            // The schedulers were told the new number and asked to switch their policy before
            // the first device was: both go back, and then the generation's graphs.
            self.record_generation(self.state.generations.current());
            self.rollback_policy();
            self.discard_devices(left);
            let report = self.state.rolled_back(failure.reason());
            self.send_report(report);
            return Ok(None);
        }
        // The trial begins the moment every device executes the new generation, and the captures
        // are the last part of the switch: a capture that fails rolls back a generation that
        // *executed* - a reverse switch on every device - and not one that was only prepared.
        self.state
            .switch(generation)
            .context("recording the switch every device made")?;
        self.capture(generation)
    }

    /// Moves the schedulers' policy and every device to `generation` at a batch boundary.
    fn cut_over(&mut self, generation: u64) -> std::result::Result<(), UpdateFailure> {
        // Before the devices switch and before a batch can be formed: the generation a step
        // stamps its batches with is the one whose graphs the runner has to find.
        self.record_generation(generation);
        let policies = self.switch_policy(generation);
        self.switching = Some(Switching {
            generation,
            captures: Vec::new(),
            policies,
            policy_deadline: Instant::now() + self.watchdog.stall_bound(),
            moved: Instant::now(),
            steps: Vec::new(),
        });
        for worker in &self.workers {
            let device = worker.device();
            let switched = worker
                .switch(generation)
                .map_err(anyhow::Error::from)
                .and_then(|submission| {
                    device_command(&self.watchdog, device, "switch", submission)
                });
            if let Err(error) = switched {
                self.switching = None;
                return Err(UpdateFailure::Device {
                    device: u32::try_from(device.0).unwrap_or_default(),
                    error: error.to_string(),
                    batch: None,
                });
            }
        }
        Ok(())
    }

    /// Queues one capture job per device for the generation that just took over.
    fn capture(&mut self, generation: u64) -> Result<Option<UpdateFailure>> {
        let mut captures = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            let ordinal = u32::try_from(worker.device().0).unwrap_or_default();
            match worker.warmup() {
                Ok(submission) => {
                    self.wake_on_submission(&submission);
                    captures.push((ordinal, submission));
                }
                Err(error) => {
                    self.switching = None;
                    return self.failure(UpdateFailure::Device {
                        device: ordinal,
                        error: format!("queueing the captures of generation {generation}: {error}"),
                        batch: None,
                    });
                }
            }
        }
        tracing::info!(
            generation,
            devices = self.workers.len(),
            "every device executes the generation; its captures are queued"
        );
        if let Some(switching) = self.switching.as_mut() {
            switching.captures = captures;
            switching.moved = Instant::now();
        }
        self.continue_switch()
    }

    /// Continues the switch in flight: reports the captures' progress, fails a capture or a
    /// policy that failed or stood still, and ends the switch once every device captured and
    /// every scheduler responded.
    ///
    /// A capture is the first run of every captured bucket of the generation that took over, so
    /// it is where a graph that cannot be made on this device surfaces - after the generation
    /// executes, which is why the failure goes through the decision table: a generation on trial
    /// rolls back, and a start fails.
    pub(super) fn continue_switch(&mut self) -> Result<Option<UpdateFailure>> {
        let Some(mut switching) = self.switching.take() else {
            return Ok(None);
        };
        let generation = switching.generation;
        self.note_capture_progress(&mut switching);

        // The schedulers' responses to the policy switch: one that refused - it holds no policy
        // a preparation made ready - fails the update rather than leaving a scheduler with no
        // policy, which forms no batch.
        let mut refused = None;
        let overdue = Instant::now() >= switching.policy_deadline;
        switching
            .policies
            .retain(|response| match response.try_recv() {
                Ok(Ok(())) => false,
                Ok(Err(error)) => {
                    refused.get_or_insert(error);
                    false
                }
                // A scheduler that stopped is reported by `watch_schedulers`.
                Err(crossbeam_channel::TryRecvError::Disconnected) => false,
                Err(crossbeam_channel::TryRecvError::Empty) if overdue => {
                    refused.get_or_insert_with(|| {
                        "a scheduler did not respond to the policy switch within the stall bound"
                            .to_owned()
                    });
                    false
                }
                Err(crossbeam_channel::TryRecvError::Empty) => true,
            });
        if let Some(error) = refused {
            tracing::error!(%error, "a scheduler's policy switch failed");
            return self.failure(UpdateFailure::Policy {
                error: format!("the policy switch failed: {error}"),
            });
        }

        let mut pending = Vec::with_capacity(switching.captures.len());
        for (device, submission) in std::mem::take(&mut switching.captures) {
            let reason = match submission.outcome() {
                None => {
                    pending.push((device, submission));
                    continue;
                }
                Some(Outcome::Success(_)) => {
                    tracing::info!(device, generation, "a device's captures finished");
                    continue;
                }
                Some(Outcome::Failed(error)) => error.to_string(),
                Some(Outcome::Cancelled) => "the captures were cancelled".to_owned(),
            };
            return self.failure(UpdateFailure::Device {
                device,
                error: format!("capturing generation {generation} on device {device}: {reason}"),
                batch: None,
            });
        }
        // A capture standing on one step past twice the stall bound is a device that will not
        // finish it: failing it by name rolls the update back and leaves the engine executing
        // what it held, where silence would be a daemon's timeout with no cause in the log.
        if let Some((device, _)) = pending.first()
            && switching.moved.elapsed() > self.watchdog.stall_bound() * 2
        {
            let device = *device;
            return self.failure(UpdateFailure::Device {
                device,
                error: format!(
                    "capturing generation {generation} on device {device} stood on one step for \
                     {}s",
                    switching.moved.elapsed().as_secs()
                ),
                batch: None,
            });
        }
        switching.captures = pending;
        if !switching.captures.is_empty() || !switching.policies.is_empty() {
            self.switching = Some(switching);
            return Ok(None);
        }

        // Every bucket a batch can be formed into exists, and every scheduler decides by the
        // generation's policy.
        if self.state.lifecycle() == Lifecycle::Starting {
            self.finish_start()?;
        } else {
            tracing::info!(
                generation,
                "the generation executes and its captures finished"
            );
            let report = self.state.switched(generation);
            self.send_report(report);
        }
        Ok(None)
    }

    /// Reports a capture that moved on since the thread last ran.
    ///
    /// What a device stands on (`Worker::performing`) changes with each capture: a change
    /// restarts this process's own bound and is sent to the daemon as `Warming`, which keeps its
    /// wait alive for as long as the captures keep moving.
    fn note_capture_progress(&mut self, switching: &mut Switching) {
        if switching.captures.is_empty() {
            return;
        }
        let steps: Vec<Option<String>> = switching
            .captures
            .iter()
            .map(|(device, _)| {
                self.workers
                    .iter()
                    .find(|worker| u32::try_from(worker.device().0).ok() == Some(*device))
                    .and_then(|worker| worker.performing())
            })
            .collect();
        if steps == switching.steps {
            return;
        }
        let moved = steps
            .iter()
            .zip(switching.steps.iter().chain(std::iter::repeat(&None)))
            .find(|(now, before)| now.is_some() && now != before)
            .and_then(|(now, _)| now.clone());
        switching.steps = steps;
        let Some(step) = moved else {
            return;
        };
        switching.moved = Instant::now();
        self.send_report(proto::process::Report {
            kind: Some(proto::process::report::Kind::Warming(
                proto::process::Warming {
                    generation: switching.generation,
                    step,
                },
            )),
        });
    }

    /// The commit: the trial ended in favour of the generation that executes, so the prepared
    /// policy is the serving one and `kept`, the generation the trial kept, can never execute
    /// again and is released.
    pub(super) fn commit(&mut self, kept: u64) {
        self.commit_policy();
        self.release(kept);
    }

    /// The rollback: the engine leaves generation `left`. One that `executed` goes back by the
    /// reverse switch on every device, to the generation the trial kept, and is then released;
    /// one that was only prepared is discarded and what serves is untouched.
    ///
    /// # Errors
    ///
    /// A device that cannot go back to the generation that executed is a device with no
    /// generation at all: the update's kernels are the ones its graphs were rebound to, and a
    /// rollback that half happened leaves the engine executing from neither. The error fails the
    /// engine by name instead of leaving it to fail on a batch that finds no graph.
    pub(super) fn roll_back(&mut self, left: u64, executed: bool) -> Result<()> {
        // The captures and the policy responses of the switch this rollback undoes decide
        // nothing any more: what runs when a queued capture reaches its device is the generation
        // that serves.
        self.switching = None;
        if !executed {
            self.discard_generation(left);
            return Ok(());
        }
        let current = self.state.generations.current();
        // The batches that form from here execute the kept generation, so the number they are
        // stamped with is that one: the update's graphs are about to be released.
        self.record_generation(current);
        self.rollback_policy();
        for worker in &self.workers {
            let device = worker.device();
            let submission = worker.rollback(current).with_context(|| {
                format!(
                    "queueing device {}'s rollback to generation {current}",
                    device.0
                )
            })?;
            device_command(&self.watchdog, device, "rollback", submission)?;
        }
        // The generation that executed and was left would otherwise keep its graphs, its
        // kernels and its arena until the process exits. Every device has finished going back,
        // so no batch of it can be in flight.
        self.release(left);
        Ok(())
    }

    /// Releases `generation`, which executed and never will again: its graphs, and the kernels
    /// whose last reference they held.
    ///
    /// The release is a command in each device's queue, behind every batch that queue already
    /// holds, so it follows the generation's last batch by the device's own order. A release
    /// that does not finish leaks the generation's storage rather than breaking correctness: it
    /// is logged and the engine serves on.
    fn release(&mut self, generation: u64) {
        for worker in &self.workers {
            let device = worker.device();
            let released = worker
                .release(generation)
                .map_err(anyhow::Error::from)
                .and_then(|submission| {
                    device_command(&self.watchdog, device, "release", submission)
                });
            if let Err(error) = released {
                tracing::error!(
                    %error,
                    generation,
                    device = device.0,
                    "releasing a generation failed; its storage stays until the process ends"
                );
            }
        }
    }
}
