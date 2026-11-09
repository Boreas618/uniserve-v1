//! The daemon's and the peers' messages: the process interface's commands, the request
//! interface's frames, and the dispatch of every program-layer command ([`ipc::ProgramCommand`])
//! to the stage that applies it, with the answer each one is owed.

use anyhow::{Result, bail};
use ipc::{ProgramCommand, ProgramStatus, UpdateFailure};

use super::MainProcess;
use super::reports::status_name;
use super::requests::{Served, engine_command_name};

/// The stage a program-layer command belongs to, for the log line each one leaves.
fn program_stage(command: &ProgramCommand) -> &'static str {
    match command {
        ProgramCommand::Precopy { .. } => "precopy",
        ProgramCommand::Switchover { .. } => "switchover",
        ProgramCommand::Commit { .. } => "commit",
        ProgramCommand::Revert { .. } => "revert",
        ProgramCommand::Purge { .. } => "purge",
    }
}

impl MainProcess {
    /// Serves the daemon's buffered messages: request submissions and commands
    /// go to the schedulers, program-layer commands to [`Self::program`].
    pub(super) fn daemon_messages(&mut self) -> Result<Option<UpdateFailure>> {
        // Every front is read in turn, the launching daemon's first: a submission names the front
        // it came from, and so does every command for a request already admitted.
        let fronts = 1 + self.control.attached_fronts();
        for front in 0..fronts {
            if front == 0 && self.daemon_closed {
                continue;
            }
            // A frame from a closed front ends that front and nothing else: its requests stay parked
            // for the takeover that claims them.
            while let Some(connection) = self.control.front_mut(front) {
                let message = match connection.try_recv() {
                    Ok(Some(message)) => message,
                    Ok(None) => break,
                    Err(error) => {
                        if front == 0 {
                            // The launching daemon leaving is the handover's own step: the
                            // attached daemons hold this process, and a front polled after it
                            // closed answers every pass with the same error - which is a log
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

    /// Serves the daemon's request interface: submissions and per-request
    /// commands are routed to the scheduler that holds the request, and
    /// program-layer commands run the program-layer update's driver.
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
                for wire in submit.requests {
                    let id = wire.id;
                    // A drained epoch and a held gate admit nothing: the refusal is retryable
                    // and the request is the frontend's to resubmit when the epoch serves again.
                    if !self.serving || self.paused {
                        self.reject(
                            front,
                            id,
                            proto::request::RejectionKind::Overloaded,
                            if self.paused {
                                "the frontend's admissions are held".to_owned()
                            } else if !self.boundary {
                                "the epoch is starting and has reached no batch boundary yet"
                                    .to_owned()
                            } else {
                                "the epoch is drained and serves nothing".to_owned()
                            },
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
                    // is a kernel program this epoch does not hold: computing the base model
                    // for a request that named an adapter would answer another model's text.
                    if let proto::request::Request::Generation(generation) = &request
                        && generation.adapter != 0
                    {
                        let adapter = generation.adapter;
                        self.reject(
                            front,
                            id,
                            proto::request::RejectionKind::Invalid,
                            format!(
                                "the adapter in slot {adapter} is loaded, and no kernel program of \
                                 this epoch computes with an adapter"
                            ),
                        );
                        continue;
                    }
                    let rank = self.route(&request);
                    let Some(scheduler) = self.schedulers.get(rank) else {
                        bail!("a request was routed to rank {rank}, which holds no scheduler");
                    };
                    match scheduler.submit(request) {
                        Ok(stream) => {
                            self.streams.push(Served {
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
                Ok(None)
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

    /// Serves one command of the daemon's process interface.
    ///
    /// `ipc::ProgramCommand` is the one form the driver reads, whether the command came from the
    /// daemon's Protobuf or from a peer of this epoch, so the conversion happens here and nowhere
    /// else. The daemon's `Commit` and `Revert` carry no generation of their own; the one they
    /// name is the generation this process is serving.
    pub(super) fn process_command(
        &mut self,
        command: proto::process::Command,
    ) -> Result<Option<UpdateFailure>> {
        let current = self.update.current();
        // A revert carries the daemon's own account of why, which `ipc::ProgramCommand` does not
        // keep: read it here so the report repeats the operator's reason rather than inventing one.
        let reverted_by = match &command.kind {
            Some(proto::process::command::Kind::Revert(revert)) => {
                Some(revert.reason.clone()).filter(|reason| !reason.is_empty())
            }
            _ => None,
        };
        let Some(program) = ProgramCommand::from_command(command.clone(), current)? else {
            // The rest are this process's own lifecycle, and no scheduler's: sched's handle
            // answers every one of them with exactly that (a park, promote, drain, serve or pause
            // is the engine process's own command). Forwarding them there and ending the epoch on
            // the refusal is how a drain used to end the epoch it was meant to keep alive.
            match command.kind {
                Some(proto::process::command::Kind::Drain(drain)) => self.drain(drain.epoch)?,
                Some(proto::process::command::Kind::Serve(serve)) => self.resume(serve)?,
                Some(proto::process::command::Kind::Pause(pause)) => {
                    self.paused = pause.paused;
                }
                // A standby parks and is then promoted in place. Both commands park and
                // resume the schedulers, and each has an answer the daemon's epoch layer waits
                // for: it promotes a child only after `Parked`, and switches the frontend only
                // after `Promoted`.
                Some(proto::process::command::Kind::Park(_)) => {
                    self.park_schedulers("parking for a promotion")?;
                    self.send_report(proto::process::Report {
                        kind: Some(proto::process::report::Kind::Parked(
                            proto::process::Parked {
                                // The daemon's epoch layer promotes a child by its
                                // launch generation: this process's own number, not
                                // the program generation (which is zero until a
                                // `Precopy` names one).
                                generation: self.control.engine(),
                            },
                        )),
                    });
                }
                Some(proto::process::command::Kind::Promote(_)) => {
                    self.promote_in_place()?;
                }
                // The five program commands were converted above; a command that states no kind
                // states nothing to apply.
                _ => {}
            }
            return Ok(None);
        };
        match self.program_with(program, None, reverted_by)? {
            // A precopy answers when its installs finish, from `advance_precopy`.
            None => Ok(None),
            Some(status) => {
                self.report(status);
                Ok(None)
            }
        }
    }

    /// Serves one peer's program-layer messages: the scheduler node's fan-out
    /// of the precopy and the commands of the switchover, commit, revert and purge.
    pub(super) fn peer_messages(&mut self, rank: u32) -> Result<Option<UpdateFailure>> {
        loop {
            let Some(command) = ({
                let Some(channel) = self.control.peer_mut(rank) else {
                    return Ok(None);
                };
                channel.try_recv::<ProgramCommand>()?
            }) else {
                return Ok(None);
            };
            if let Some(status) = self.program(command, Some(rank))? {
                self.answer(Some(rank), status);
            }
        }
    }

    /// One program-layer command.
    ///
    /// A precopy answers later, from [`Self::advance_precopy`], because its
    /// module builds run between serve passes; every other stage answers at
    /// once.
    fn program(
        &mut self,
        command: ProgramCommand,
        reply: Option<u32>,
    ) -> Result<Option<ProgramStatus>> {
        self.program_with(command, reply, None)
    }

    /// One program-layer command, with the daemon's own reason when it stated one.
    ///
    /// A revert's reason is the daemon's account of why an operator ended the probation; it is
    /// carried into the epoch's record instead of a reason this process makes up. A peer's command
    /// states none, and the epoch's own failures state their own.
    fn program_with(
        &mut self,
        command: ProgramCommand,
        reply: Option<u32>,
        reverted_by: Option<String>,
    ) -> Result<Option<ProgramStatus>> {
        // The program layer's stages are the epoch's steps, and a step that never arrives is the
        // failure this line is for: it names the command and the generation, so a log that stops
        // after "claimed the deployment's storage" says which step the process is waiting for.
        self.commands_seen += 1;
        tracing::info!(
            stage = program_stage(&command),
            boundary = self.boundary,
            serving = self.serving,
            warming = self.warming.len(),
            "a program-layer command arrived"
        );
        let status = match command {
            ProgramCommand::Precopy {
                generation,
                programs,
                policy,
            } => {
                // The receipt: this line says the daemon's step reached this process and what it
                // carried, before anything is resolved, built or installed. A run that stops with
                // no such line stopped before the daemon sent it, and one that stops after it
                // stopped here.
                tracing::info!(
                    generation,
                    operators = programs.len(),
                    policy = %policy.backend,
                    policy_revision = policy.revision,
                    "the daemon's Precopy arrived"
                );
                self.start_precopy(generation, &programs, &policy, reply)?;
                return Ok(None);
            }
            ProgramCommand::Switchover { generation, step } => {
                tracing::info!(
                    generation,
                    step = step.unwrap_or_default(),
                    "received the daemon's switchover"
                );
                match self.switchover(generation, step, reply)? {
                    Some(status) => status,
                    // The answer is owed to the warmup: it is sent from `advance_warmup` when
                    // admission opens, or replaced by the failure that keeps admission shut.
                    None => return Ok(None),
                }
            }
            ProgramCommand::Commit { generation } => self.commit(generation)?,
            // The operator's rollback is the daemon's decision, and its reason - when the
            // daemon states one - is what the record keeps. It is not a policy failure: nothing
            // about the scheduling policy failed, and saying it did would put a policy error in
            // the epoch's record and in the daemon's account of why the update ended.
            ProgramCommand::Revert { generation } => self.revert(
                generation,
                UpdateFailure::Threshold {
                    reason: reverted_by.unwrap_or_else(|| "the daemon reverted the update".into()),
                },
            )?,
            ProgramCommand::Purge { generation } => self.purge(generation)?,
        };
        Ok(Some(status))
    }

    /// Delivers a program-layer status: to the peer that asked, and to the
    /// daemon.
    pub(super) fn answer(&mut self, reply: Option<u32>, status: ProgramStatus) {
        // The program layer's steps are answered here. A run that stops between two of them needs
        // to know which one was answered last, and the daemon's own log of its commands is not this
        // side's. The reason travels with the status, and a run that reads only this log needs it
        // too.
        let failure = match &status {
            ProgramStatus::PrecopyFailed { failure, .. } => format!("{failure:?}"),
            _ => String::new(),
        };
        tracing::info!(
            status = status_name(&status),
            to_peer = reply.is_some(),
            failure,
            "answering the epoch's program layer"
        );
        if let Some(rank) = reply
            && let Some(channel) = self.control.peer_mut(rank)
            && let Err(error) = channel.send(&status)
        {
            tracing::warn!(%error, rank, "answering a peer's program command failed");
        }
        self.report(status);
    }
}
