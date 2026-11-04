//! The peer messages of a program-layer update (`specs/worker.md` §2.3, §7.3).
//!
//! The daemon receives `PUT /admin/config`, decides that the two configurations
//! differ only in the programs they fill each operator with (§7.2), and prepares
//! the target programs at each host's Program Server. The update itself then
//! runs between the main processes of the epoch:
//!
//! 1. the scheduler node's main process receives `Precopy` and fans it out to
//!    every main process of the epoch over the existing ipc channel (§2.3), so
//!    that all ranks install the same generation;
//! 2. every main process answers `Precopied` or `PrecopyFailed` (step 2);
//! 3. after all have answered, the daemon sends `Switchover` to each DP rank's
//!    scheduler, which puts it in its batch sequence at a batch boundary and
//!    tells its ranks (step 3);
//! 4. during probation, `Commit` or `Revert` ends the update (step 5), and a
//!    failure a main process finds is reported as [`crate::ProgramStatus::Failed`]
//!    or [`crate::ProgramStatus::PolicyReverted`];
//! 5. `Purge { generation }` releases the generation before the serving one once
//!    its last batch has completed (step 6).
//!
//! The messages carry a program's names — its backend, its revision and its
//! arguments — and never its content: content comes from the Program Server by
//! `resolve` (§6.3). `Program`, `ProgramSlot`, `PhaseKind` and `Programs` are
//! defined in `crates/program-server` (design §5.1); this module gives them their
//! form on the peer channel, exactly as the daemon's append-only Protobuf
//! carries them across builds.
//!
//! This module is declared in `crates/ipc/src/lib.rs` when `program-server` is a
//! workspace member here. Until then the commands cannot be compiled, because
//! their payload's owners are not in the tree; the statuses they answer with are
//! in [`crate::update`] and compile today.

use program_server::program::Programs;
use serde::{Deserialize, Serialize};

/// A command of the program-layer update, as a main process receives it.
///
/// The daemon-to-main-process form is the append-only Protobuf of `crates/proto`
/// (§2.3); this is the same command between main processes of one epoch, which
/// are one build and therefore exchange Rust types.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramCommand {
    /// §7.3 step 2: install `programs` as generation `generation` while serving
    /// goes on.
    Precopy {
        /// The generation being installed: one update number, assigned by the
        /// daemon and shared by every main process of the epoch.
        generation: u64,
        /// The operators' programs to install, and the programs they replace
        /// (kept through probation, §7.3 step 4).
        programs: Box<Programs>,
        /// The scheduling policy the same update installs. It is not a slot of
        /// `programs`: a policy is one program per epoch, named by its own
        /// backend, and the daemon's epoch record keeps the two apart
        /// (`crates/daemon/src/epoch.rs`).
        policy: program_server::Program,
    },
    /// §7.3 step 3: cut over at the batch boundary whose step is `step`.
    Switchover {
        /// The generation to serve.
        generation: u64,
        /// The scheduler step the switchover runs at, agreed by an expert
        /// group's schedulers so that both ends of an all-to-all switch
        /// together.
        ///
        /// `None` when the command came from the daemon, which does not choose
        /// the boundary: the receiving main process's scheduler names the step
        /// at its next batch boundary. `Some` between main processes of one
        /// epoch, whose schedulers agreed on it.
        step: Option<u64>,
    },
    /// §7.3 step 5: probation ended in a commit.
    Commit {
        /// The generation committed.
        generation: u64,
    },
    /// §7.3 step 4: revert to the generation serving before the update.
    Revert {
        /// The generation reverted away from.
        generation: u64,
    },
    /// §7.3 step 6: release generation `generation`, which is older than the one
    /// serving.
    Purge {
        /// The generation to purge.
        generation: u64,
    },
}

impl ProgramCommand {
    /// The generation this command is about.
    pub const fn generation(&self) -> u64 {
        match self {
            Self::Precopy { generation, .. }
            | Self::Switchover { generation, .. }
            | Self::Commit { generation }
            | Self::Revert { generation }
            | Self::Purge { generation } => *generation,
        }
    }

    /// The stage's name, for tracing and for the status the command answers with.
    pub const fn stage(&self) -> &'static str {
        match self {
            Self::Precopy { .. } => "precopy",
            Self::Switchover { .. } => "switchover",
            Self::Commit { .. } => "commit",
            Self::Revert { .. } => "revert",
            Self::Purge { .. } => "purge",
        }
    }
}

/// These commands cross two wires with one shape: the daemon's Protobuf
/// (`proto::process::Command`, §2.3) and the peer channel of one epoch's main
/// processes, which are one build and exchange this Rust type. The mapping is
/// written once, here, so a field cannot be carried by one wire and dropped by
/// the other without a reader seeing it.
impl ProgramCommand {
    /// This command as the daemon's Protobuf `Command`.
    ///
    /// The switchover's `step` is not part of the daemon's form: the boundary
    /// is the main process's own decision (§7.3 step 3), and the daemon learns
    /// it from `Report::Switched`.
    pub fn to_command(&self) -> proto::process::Command {
        use proto::process::command::Kind;

        let kind = match self {
            Self::Precopy { programs, policy, .. } => Kind::Precopy(proto::process::Precopy {
                kernels: programs
                    .iter()
                    .map(|(slot, program)| proto::process::KernelProgram {
                        op: slot.op.clone(),
                        phase: slot.phase.map(phase_to_wire).map(i32::from),
                        program: Some(program_to_wire(program)),
                    })
                    .collect(),
                policy: Some(program_to_wire(policy)),
            }),
            Self::Switchover { generation, .. } => {
                Kind::Switchover(proto::process::Switchover {
                    generation: *generation,
                })
            }
            Self::Commit { .. } => Kind::Commit(proto::process::Commit {}),
            Self::Revert { .. } => Kind::Revert(proto::process::Revert {
                reason: String::new(),
            }),
            Self::Purge { generation } => Kind::Purge(proto::process::Purge {
                generation: *generation,
            }),
        };
        proto::process::Command { kind: Some(kind) }
    }

    /// The program-layer command `command` states, for a main process serving
    /// generation `current`.
    ///
    /// `None` for the daemon's other commands (park, promote, drain, serve,
    /// pause), which are the epoch layer's and not the program layer's. The
    /// daemon's commit and revert carry no generation - it is the one serving
    /// when they arrive - so the state machine's number is `current` (for a
    /// revert, the generation being reverted away from).
    pub fn from_command(
        command: proto::process::Command,
        current: u64,
    ) -> Result<Option<Self>, crate::IpcError> {
        use proto::process::command::Kind;

        let Some(kind) = command.kind else {
            return Err(program_error("a command states no kind"));
        };
        Ok(Some(match kind {
            Kind::Precopy(precopy) => {
                let mut programs = Programs::new();
                for kernel in precopy.kernels {
                    let program = kernel
                        .program
                        .ok_or_else(|| program_error("a precopy's kernel states no program"))?;
                    let phase = kernel
                        .phase
                        .map(phase_from_wire)
                        .transpose()?
                        .flatten();
                    let slot = program_server::ProgramSlot {
                        op: kernel.op,
                        phase,
                    };
                    programs.insert(slot, program_from_wire(program)?);
                }
                let policy = precopy
                    .policy
                    .ok_or_else(|| program_error("a precopy states no policy"))?;
                Self::Precopy {
                    // The daemon's precopy states no number: the main process
                    // numbers the generation it prepares (design 7.3 step 2),
                    // and `Report::Precopied` echoes it back.
                    generation: current + 1,
                    programs: Box::new(programs),
                    policy: program_from_wire(policy)?,
                }
            }
            Kind::Switchover(switchover) => Self::Switchover {
                generation: switchover.generation,
                step: None,
            },
            Kind::Commit(_) => Self::Commit { generation: current },
            Kind::Revert(_) => Self::Revert { generation: current },
            Kind::Purge(purge) => Self::Purge {
                generation: purge.generation,
            },
            _ => return Ok(None),
        }))
    }
}

/// A program as the daemon's Protobuf carries it: its backend, its revision and
/// its arguments, never the module's bytes (§6.3).
fn program_to_wire(program: &program_server::Program) -> proto::process::Program {
    proto::process::Program {
        backend: program.backend.clone(),
        revision: program.revision,
        arguments: program
            .arguments
            .iter()
            .map(|(name, value)| (name.clone(), value.to_string()))
            .collect(),
    }
}

/// The program a Protobuf form states, its arguments read back as TOML values.
fn program_from_wire(
    program: proto::process::Program,
) -> Result<program_server::Program, crate::IpcError> {
    let mut arguments = toml::Table::new();
    for (name, value) in program.arguments {
        let value = value.parse::<toml::Value>().map_err(|_| {
            program_error(format!(
                "the argument {name} of {} is not a TOML value",
                program.backend
            ))
        })?;
        arguments.insert(name, value);
    }
    Ok(program_server::Program {
        backend: program.backend,
        revision: program.revision,
        arguments,
    })
}

/// `ops::PhaseKind` as the wire's enum.
fn phase_to_wire(phase: program_server::PhaseKind) -> proto::process::PhaseKind {
    use program_server::PhaseKind;
    match phase {
        PhaseKind::Prefill => proto::process::PhaseKind::Prefill,
        PhaseKind::Decode => proto::process::PhaseKind::Decode,
        PhaseKind::Encode => proto::process::PhaseKind::Encode,
        PhaseKind::Denoise => proto::process::PhaseKind::Denoise,
    }
}

/// The wire's enum as `ops::PhaseKind`; `Ok(None)` for the unspecified value,
/// which means the program fills every phase of its operator.
fn phase_from_wire(
    phase: i32,
) -> Result<Option<program_server::PhaseKind>, crate::IpcError> {
    use proto::process::PhaseKind;
    Ok(match PhaseKind::try_from(phase) {
        Ok(PhaseKind::Prefill) => Some(program_server::PhaseKind::Prefill),
        Ok(PhaseKind::Decode) => Some(program_server::PhaseKind::Decode),
        Ok(PhaseKind::Encode) => Some(program_server::PhaseKind::Encode),
        Ok(PhaseKind::Denoise) => Some(program_server::PhaseKind::Denoise),
        Ok(PhaseKind::Unspecified) => None,
        Err(_) => return Err(program_error(format!("phase {phase} is not a phase kind"))),
    })
}

/// A program-wire failure.
fn program_error(what: impl Into<std::borrow::Cow<'static, str>>) -> crate::IpcError {
    crate::EncodingError::Program { what: what.into() }.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_command_names_its_generation_and_stage() {
        let command = ProgramCommand::Switchover {
            generation: 4,
            step: Some(17),
        };
        assert_eq!(command.generation(), 4);
        assert_eq!(command.stage(), "switchover");
    }

    #[test]
    fn a_precopy_carries_program_names_and_arguments_and_never_content() {
        let mut programs = Programs::new();
        programs.insert(
            program_server::ProgramSlot::all("decode_attention"),
            program_server::Program {
                backend: "trtllm".to_owned(),
                revision: 3,
                arguments: "block = 16".parse().expect("the program's arguments"),
            },
        );
        programs.insert(
            program_server::ProgramSlot::phase(
                "rms_norm",
                program_server::PhaseKind::Prefill,
            ),
            program_server::Program {
                backend: "triton".to_owned(),
                revision: 2,
                arguments: toml::Table::new(),
            },
        );
        let command = ProgramCommand::Precopy {
            generation: 5,
            programs: Box::new(programs.clone()),
            policy: program_server::Program {
                backend: "fcfs".to_owned(),
                revision: 1,
                arguments: "chunked_prefill_size = 4096".parse().expect("the policy's arguments"),
            },
        };

        let encoded = rmp_serde::to_vec(&command).expect("a precopy encodes");
        // The module's bytes are not part of the command: a main process
        // resolves them from its Program Server (§6.3), and the peer channel
        // carries names only.
        assert!(
            encoded.len() < 1024,
            "a precopy of two programs is {} bytes",
            encoded.len()
        );
        let decoded: ProgramCommand = rmp_serde::from_slice(&encoded).expect("and decodes");
        assert_eq!(decoded, command);
        let ProgramCommand::Precopy {
            generation,
            programs,
            policy,
        } = decoded
        else {
            panic!("a precopy decoded as {}", decoded.stage());
        };
        assert_eq!(generation, 5);
        assert_eq!(programs[&program_server::ProgramSlot::all("decode_attention")].revision, 3);
        assert_eq!(policy.backend, "fcfs");
    }

    #[test]
    fn the_commands_between_main_processes_survive_the_peer_encoding() {
        for command in [
            ProgramCommand::Switchover {
                generation: 4,
                step: Some(17),
            },
            ProgramCommand::Commit { generation: 4 },
            ProgramCommand::Revert { generation: 4 },
            ProgramCommand::Purge { generation: 4 },
        ] {
            let encoded = rmp_serde::to_vec(&command).expect("a program command encodes");
            let decoded: ProgramCommand =
                rmp_serde::from_slice(&encoded).expect("and decodes");
            assert_eq!(decoded, command);
        }
    }

    #[test]
    fn a_precopy_survives_the_daemon_wire_with_its_policy_and_arguments() {
        let mut programs = Programs::new();
        programs.insert(
            program_server::ProgramSlot::phase("rms_norm", program_server::PhaseKind::Prefill),
            program_server::Program {
                backend: "triton".to_owned(),
                revision: 2,
                arguments: "block = 256".parse().expect("the program's arguments"),
            },
        );
        let command = ProgramCommand::Precopy {
            generation: 9,
            programs: Box::new(programs),
            policy: program_server::Program {
                backend: "fcfs".to_owned(),
                revision: 4,
                arguments: "chunked_prefill_size = 4096".parse().expect("the policy's arguments"),
            },
        };

        let wire = command.to_command();
        let back = ProgramCommand::from_command(wire.clone(), 3)
            .expect("the wire form states a program command")
            .expect("and it is the program layer's");
        assert_eq!(back, command);

        // The daemon's form carries no modules, only names, revisions and
        // arguments (§6.3), and no step: the boundary is the main process's.
        let bytes = prost::Message::encoded_len(&wire);
        assert!(bytes < 512, "a precopy of one kernel is {bytes} bytes");
        assert_eq!(
            command.to_command(),
            ProgramCommand::from_command(wire, 3)
                .expect("decodes")
                .expect("is a program command")
                .to_command()
        );
    }

    #[test]
    fn the_daemons_commit_and_revert_name_the_generation_the_process_serves() {
        use proto::process::command::Kind;

        for kind in [
            Kind::Commit(proto::process::Commit {}),
            Kind::Revert(proto::process::Revert {
                reason: String::new(),
            }),
        ] {
            let command = ProgramCommand::from_command(proto::process::Command { kind: Some(kind) }, 7)
                .expect("the wire form states a program command")
                .expect("and it is the program layer's");
            assert_eq!(command.generation(), 7);
        }
    }

    #[test]
    fn a_command_outside_the_program_layer_is_none() {
        use proto::process::command::Kind;

        for kind in [
            Kind::Park(proto::process::Park {}),
            Kind::Promote(proto::process::Promote {}),
            Kind::Drain(proto::process::Drain { epoch: 2 }),
            Kind::Pause(proto::process::Pause { paused: true }),
        ] {
            let command = ProgramCommand::from_command(proto::process::Command { kind: Some(kind) }, 1)
                .expect("the command is well formed");
            assert!(command.is_none());
        }
    }

}
