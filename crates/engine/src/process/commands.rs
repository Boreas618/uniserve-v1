//! The daemon's messages: the request interface's frames, and the process interface's commands,
//! each applied to the engine's state ([`super::lifecycle::State::command`]) and then carried out
//! as the step that state returns.

use anyhow::{Result, bail};
use ipc::{GenerationCommand, UpdateFailure};
use proto::process::command;

use super::MainProcess;
use super::lifecycle::Step;
use super::requests::{RequestStream, engine_command_name};

impl MainProcess {
    /// Serves the daemon's buffered messages: request submissions and per-request commands go
    /// to the schedulers, engine commands to [`Self::engine_command`].
    pub(super) fn daemon_messages(&mut self) -> Result<Option<UpdateFailure>> {
        // Every front is read in turn, the launching daemon's first: a submission names the front
        // it came from, and so does every command for a request already admitted.
        let fronts = 1 + self.control.attached_fronts();
        for front in 0..fronts {
            if front == 0 && self.daemon_closed {
                continue;
            }
            // A frame from a closed front ends that front and nothing else: its requests stay
            // for the takeover that takes them.
            while let Some(connection) = self.control.front_mut(front) {
                let message = match connection.try_recv() {
                    Ok(Some(message)) => message,
                    Ok(None) => break,
                    Err(error) => {
                        if front == 0 {
                            // The launching daemon leaving is the handover's own step: the
                            // attached daemons hold this process, and a front polled after it
                            // closed gives every read the same error - which is a log
                            // that hides the handover's own lines behind it.
                            self.daemon_closed = true;
                            tracing::warn!(
                                %error,
                                "the daemon that launched this process closed its front; the \
                                 attached daemons serve on, so this front is not read again"
                            );
                        } else {
                            tracing::warn!(%error, front, "reading a daemon front failed");
                            self.control.detach(front);
                        }
                        break;
                    }
                };
                if let Some(failure) = self.daemon_message(front, message)? {
                    return Ok(Some(failure));
                }
            }
        }
        Ok(None)
    }

    /// Serves one frame of the daemon's request interface: a submission and a per-request
    /// command are routed to the scheduler that holds the request, and an engine command to
    /// what serves it.
    fn daemon_message(
        &mut self,
        front: usize,
        message: proto::request::wire::ToEngine,
    ) -> Result<Option<UpdateFailure>> {
        let Some(kind) = message.kind else {
            return Ok(None);
        };
        match kind {
            proto::request::wire::to_engine::Kind::Submit(submit) => {
                // Every request of one submission is queued before the next
                // batch is formed.
                // A routing policy that failed with no previous policy to take over fails the
                // update once the submission's requests are responded to.
                let mut routing_failure = None;
                for wire in submit.requests {
                    let id = wire.id;
                    // Only a serving engine whose gate is open admits: the refusal is retryable
                    // and the request is the daemon's to resubmit when an engine serves.
                    let refusal = if self.paused {
                        Some("admissions are held")
                    } else {
                        self.state.lifecycle().refusal()
                    };
                    if let Some(refusal) = refusal {
                        self.reject(
                            front,
                            id,
                            proto::request::RejectionKind::Overloaded,
                            refusal.to_owned(),
                        );
                        continue;
                    }
                    let request = match proto::request::Request::from_wire(wire) {
                        Ok(request) => request,
                        Err(error) => {
                            // A request that does not convert is the daemon's, not this process's:
                            // it ends with the refusal's event and every other request of the
                            // submission is still admitted.
                            self.reject(
                                front,
                                id,
                                proto::request::RejectionKind::Invalid,
                                error.to_string(),
                            );
                            continue;
                        }
                    };
                    // An adapter's tensors load into their slot, and what computes with them
                    // is a kernel package this epoch does not hold: computing the base model
                    // for a request that named an adapter would respond with another model's text.
                    if let proto::request::Request::Generation(generation) = &request
                        && generation.adapter != 0
                    {
                        let adapter = generation.adapter;
                        self.reject(
                            front,
                            id,
                            proto::request::RejectionKind::Invalid,
                            format!(
                                "the adapter in slot {adapter} is loaded, and no kernel package of \
                                 this epoch computes with an adapter"
                            ),
                        );
                        continue;
                    }
                    let rank = match self.route(id, &request) {
                        Ok(rank) => rank,
                        Err((reason, failure)) => {
                            self.reject(
                                front,
                                id,
                                proto::request::RejectionKind::Overloaded,
                                reason,
                            );
                            if let Some(failure) = failure {
                                routing_failure.get_or_insert(failure);
                            }
                            continue;
                        }
                    };
                    let Some(scheduler) = self.schedulers.get(rank) else {
                        bail!("a request was routed to rank {rank}, which holds no scheduler");
                    };
                    match scheduler.submit(request) {
                        Ok(stream) => {
                            self.streams.push(RequestStream {
                                request: id,
                                rank,
                                events: stream,
                                front,
                                sent: 0,
                                kept: std::collections::VecDeque::new(),
                            });
                            self.send_event(
                                front,
                                proto::request::wire::from_engine::Kind::Accepted(
                                    proto::request::wire::Accepted {
                                        request: id,
                                        scheduler: rank as u32,
                                    },
                                ),
                            );
                        }
                        Err(error) => {
                            let kind = match error {
                                scheduler::SubmitError::Closed | scheduler::SubmitError::Dead => {
                                    proto::request::RejectionKind::Overloaded
                                }
                            };
                            self.reject(front, id, kind, error.to_string());
                        }
                    }
                }
                match routing_failure {
                    Some(failure) => self.failure(failure),
                    None => Ok(None),
                }
            }
            proto::request::wire::to_engine::Kind::Command(command) => {
                self.request_command(front, command)?;
                Ok(None)
            }
            proto::request::wire::to_engine::Kind::Engine(command) => {
                // What a daemon commands of the engine arrives on the connection it made, and
                // a handover's attach is the one such command whose silence reads as a hang.
                // Stated where it is read, so "the engine never saw it" and "the engine
                // refused it" are different lines in a run.
                tracing::info!(
                    front,
                    command = command.kind.as_ref().map_or("none", engine_command_name),
                    "the engine read an engine command"
                );
                self.engine_command(front, command)
            }
        }
    }

    /// Serves one command of the daemon's process interface: the engine's state decides what
    /// the command means now, and the step it returns is carried out here.
    pub(super) fn process_command(
        &mut self,
        command: proto::process::Command,
    ) -> Result<Option<UpdateFailure>> {
        let Some(kind) = command.kind else {
            return Ok(None);
        };
        let step = self.state.command(&kind);
        tracing::info!(
            command = command_name(&kind),
            ?step,
            lifecycle = ?self.state.lifecycle(),
            update = ?self.state.update(),
            "a command of the daemon's process interface arrived"
        );
        match (step, kind) {
            (Step::Prepare { generation }, kind @ command::Kind::Prepare(_)) => {
                // The one conversion of the daemon's Protobuf into the packages it names.
                let prepare = GenerationCommand::from_command(
                    proto::process::Command { kind: Some(kind) },
                    generation - 1,
                );
                match prepare {
                    Ok(Some(GenerationCommand::Prepare {
                        packages, policy, ..
                    })) => return self.start_prepare(generation, &packages, &policy),
                    Ok(_) => bail!("a prepare did not read as one"),
                    Err(error) => {
                        let report = self.state.prepare_failed(generation, error.to_string());
                        self.send_report(report);
                    }
                }
            }
            (Step::Switch { generation }, _) => return self.switch(generation),
            (Step::Commit { kept }, _) => {
                self.commit(kept);
                let report = self.state.committed();
                self.send_report(report);
            }
            (Step::Rollback { left, executed }, command::Kind::Rollback(rollback)) => {
                self.roll_back(left, executed)?;
                // The reason is the daemon's account of why the trial ended, and the
                // acknowledgement repeats it rather than inventing one.
                let reason = if rollback.reason.is_empty() {
                    "the daemon rolled the update back".to_owned()
                } else {
                    rollback.reason
                };
                let report = self.state.rolled_back(reason);
                self.send_report(report);
            }
            (Step::Drain { epoch, discarded }, _) => self.drain(epoch, discarded)?,
            (Step::Promote, command::Kind::Promote(promote)) => self.promote(promote.handoff)?,
            (Step::Pause(paused), _) => self.pause(paused),
            (Step::Report(report), _) => self.send_report(report),
            (Step::Ignore, _) => {}
            // A step is returned for the command that asked for it.
            (step, kind) => bail!(
                "the step {step:?} does not belong to the command {}",
                command_name(&kind)
            ),
        }
        Ok(None)
    }
}

/// The name of a process-interface command, for the line each one leaves.
fn command_name(kind: &command::Kind) -> &'static str {
    match kind {
        command::Kind::Prepare(_) => "prepare",
        command::Kind::Switch(_) => "switch",
        command::Kind::Commit(_) => "commit",
        command::Kind::Rollback(_) => "rollback",
        command::Kind::Drain(_) => "drain",
        command::Kind::Promote(_) => "promote",
        command::Kind::Pause(_) => "pause",
        command::Kind::Serve(_) => "serve",
    }
}
