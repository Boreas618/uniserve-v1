//! The peer messages of a generation update.
//!
//! The daemon receives `PUT /admin/config`, decides that the two configurations
//! differ only in the packages they fill each operator with (§7.2), and prepares
//! the target packages at each host's Build Server. The update itself then
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
//!    failure a main process finds is reported as [`crate::GenerationStatus::Failed`]
//!    or [`crate::GenerationStatus::PolicyReverted`];
//! 5. `Purge { generation }` releases the generation before the serving one once
//!    its last batch has completed (step 6).
//!
//! The messages carry a package's names — its backend, its revision and its
//! arguments — and never its content: content comes from the Build Server by
//! `resolve` (§6.3). `PackageRevision`, `ModuleSlot`, `PhaseKind` and `Packages` are
//! the module vocabulary of `proto::module` (design §5.1); this module gives
//! them their form on the peer channel, exactly as the daemon's append-only
//! Protobuf carries them across builds.

use proto::module::Packages;
use serde::{Deserialize, Serialize};

/// A command of the generation update, as a main process receives it.
///
/// The daemon-to-main-process form is the append-only Protobuf of `crates/proto`
/// (§2.3); this is the same command between main processes of one epoch, which
/// are one build and therefore exchange Rust types.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationCommand {
    /// §7.3 step 2: install `packages` as generation `generation` while serving
    /// goes on.
    Precopy {
        /// The generation being installed: one update number, assigned by the
        /// daemon and shared by every main process of the epoch.
        generation: u64,
        /// The operators' packages to install, and the packages they replace
        /// (kept through probation, §7.3 step 4).
        packages: Box<Packages>,
        /// The scheduling policy the same update installs. It is not a slot of
        /// `packages`: a policy is one package per epoch, named by its own
        /// backend, and the daemon's epoch record keeps the two apart
        /// (`crates/daemon/src/epoch.rs`).
        policy: proto::module::PackageRevision,
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

impl GenerationCommand {
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
impl GenerationCommand {
    /// This command as the daemon's Protobuf `Command`.
    ///
    /// The switchover's `step` is not part of the daemon's form: the boundary
    /// is the main process's own decision (§7.3 step 3), and the daemon learns
    /// it from `Report::Switched`.
    pub fn to_command(&self) -> proto::process::Command {
        use proto::process::command::Kind;

        let kind = match self {
            Self::Precopy {
                packages, policy, ..
            } => Kind::Precopy(proto::process::Precopy {
                kernels: packages
                    .iter()
                    .map(|(slot, package)| proto::process::KernelPackage {
                        op: slot.op.clone(),
                        phase: slot.phase.map(phase_to_wire).map(i32::from),
                        package: Some(package_to_wire(package)),
                        part: slot.part.clone(),
                    })
                    .collect(),
                policy: Some(package_to_wire(policy)),
            }),
            Self::Switchover { generation, .. } => Kind::Switchover(proto::process::Switchover {
                generation: *generation,
            }),
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

    /// The generation command `command` states, for a main process serving
    /// generation `current`.
    ///
    /// `None` for the daemon's other commands (park, promote, drain, serve,
    /// pause), which are the epoch layer's and not the generation layer's. The
    /// daemon's commit and revert carry no generation - it is the one serving
    /// when they arrive - so the state machine's number is `current` (for a
    /// revert, the generation being reverted away from).
    pub fn from_command(
        command: proto::process::Command,
        current: u64,
    ) -> Result<Option<Self>, crate::IpcError> {
        use proto::process::command::Kind;

        let Some(kind) = command.kind else {
            return Err(command_error("a command states no kind"));
        };
        Ok(Some(match kind {
            Kind::Precopy(precopy) => {
                let mut packages = Packages::new();
                for kernel in precopy.kernels {
                    let package = kernel
                        .package
                        .ok_or_else(|| command_error("a precopy's kernel states no package"))?;
                    let phase = kernel.phase.map(phase_from_wire).transpose()?.flatten();
                    let slot = proto::module::ModuleSlot {
                        op: kernel.op,
                        phase,
                        part: kernel.part,
                    };
                    packages.insert(slot, package_from_wire(package)?);
                }
                let policy = precopy
                    .policy
                    .ok_or_else(|| command_error("a precopy states no policy"))?;
                Self::Precopy {
                    // The daemon's precopy states no number: the main process
                    // numbers the generation it prepares (design 7.3 step 2),
                    // and `Report::Precopied` echoes it back.
                    generation: current + 1,
                    packages: Box::new(packages),
                    policy: package_from_wire(policy)?,
                }
            }
            Kind::Switchover(switchover) => Self::Switchover {
                generation: switchover.generation,
                step: None,
            },
            Kind::Commit(_) => Self::Commit {
                generation: current,
            },
            Kind::Revert(_) => Self::Revert {
                generation: current,
            },
            Kind::Purge(purge) => Self::Purge {
                generation: purge.generation,
            },
            _ => return Ok(None),
        }))
    }
}

/// A package as the daemon's Protobuf carries it: its backend, its revision and
/// its arguments, never the module's bytes (§6.3).
fn package_to_wire(package: &proto::module::PackageRevision) -> proto::process::PackageRevision {
    proto::process::PackageRevision {
        backend: package.backend.clone(),
        revision: package.revision,
        arguments: package
            .arguments
            .iter()
            .map(|(name, value)| (name.clone(), value.to_string()))
            .collect(),
    }
}

/// The package a Protobuf form states, its arguments read back as TOML values.
fn package_from_wire(
    package: proto::process::PackageRevision,
) -> Result<proto::module::PackageRevision, crate::IpcError> {
    let mut arguments = toml::Table::new();
    for (name, value) in package.arguments {
        let value = value.parse::<toml::Value>().map_err(|_| {
            command_error(format!(
                "the argument {name} of {} is not a TOML value",
                package.backend
            ))
        })?;
        arguments.insert(name, value);
    }
    Ok(proto::module::PackageRevision {
        backend: package.backend,
        revision: package.revision,
        arguments,
    })
}

/// `proto::module::PhaseKind` as the wire's enum.
fn phase_to_wire(phase: proto::module::PhaseKind) -> proto::process::PhaseKind {
    use proto::module::PhaseKind;
    match phase {
        PhaseKind::Prefill => proto::process::PhaseKind::Prefill,
        PhaseKind::Decode => proto::process::PhaseKind::Decode,
        PhaseKind::Encode => proto::process::PhaseKind::Encode,
        PhaseKind::Denoise => proto::process::PhaseKind::Denoise,
    }
}

/// The wire's enum as `proto::module::PhaseKind`; `Ok(None)` for the unspecified value,
/// which means the package fills every phase of its operator.
fn phase_from_wire(phase: i32) -> Result<Option<proto::module::PhaseKind>, crate::IpcError> {
    use proto::process::PhaseKind;
    Ok(match PhaseKind::try_from(phase) {
        Ok(PhaseKind::Prefill) => Some(proto::module::PhaseKind::Prefill),
        Ok(PhaseKind::Decode) => Some(proto::module::PhaseKind::Decode),
        Ok(PhaseKind::Encode) => Some(proto::module::PhaseKind::Encode),
        Ok(PhaseKind::Denoise) => Some(proto::module::PhaseKind::Denoise),
        Ok(PhaseKind::Unspecified) => None,
        Err(_) => return Err(command_error(format!("phase {phase} is not a phase kind"))),
    })
}

/// A failure reading a generation command from the wire.
fn command_error(what: impl Into<std::borrow::Cow<'static, str>>) -> crate::IpcError {
    crate::EncodingError::PackageRevision { what: what.into() }.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_command_names_its_generation_and_stage() {
        let command = GenerationCommand::Switchover {
            generation: 4,
            step: Some(17),
        };
        assert_eq!(command.generation(), 4);
        assert_eq!(command.stage(), "switchover");
    }

    #[test]
    fn a_precopy_carries_package_names_and_arguments_and_never_content() {
        let mut packages = Packages::new();
        packages.insert(
            proto::module::ModuleSlot::all("decode_attention"),
            proto::module::PackageRevision {
                backend: "trtllm".to_owned(),
                revision: 3,
                arguments: "block = 16".parse().expect("the package's arguments"),
            },
        );
        packages.insert(
            proto::module::ModuleSlot::phase("rms_norm", proto::module::PhaseKind::Prefill),
            proto::module::PackageRevision {
                backend: "triton".to_owned(),
                revision: 2,
                arguments: toml::Table::new(),
            },
        );
        let command = GenerationCommand::Precopy {
            generation: 5,
            packages: Box::new(packages.clone()),
            policy: proto::module::PackageRevision {
                backend: "fcfs".to_owned(),
                revision: 1,
                arguments: "chunked_prefill_size = 4096"
                    .parse()
                    .expect("the policy's arguments"),
            },
        };

        let encoded = rmp_serde::to_vec(&command).expect("a precopy encodes");
        // The module's bytes are not part of the command: a main process
        // resolves them from its Build Server (§6.3), and the peer channel
        // carries names only.
        assert!(
            encoded.len() < 1024,
            "a precopy of two packages is {} bytes",
            encoded.len()
        );
        let decoded: GenerationCommand = rmp_serde::from_slice(&encoded).expect("and decodes");
        assert_eq!(decoded, command);
        let GenerationCommand::Precopy {
            generation,
            packages,
            policy,
        } = decoded
        else {
            panic!("a precopy decoded as {}", decoded.stage());
        };
        assert_eq!(generation, 5);
        assert_eq!(
            packages[&proto::module::ModuleSlot::all("decode_attention")].revision,
            3
        );
        assert_eq!(policy.backend, "fcfs");
    }

    #[test]
    fn the_commands_between_main_processes_survive_the_peer_encoding() {
        for command in [
            GenerationCommand::Switchover {
                generation: 4,
                step: Some(17),
            },
            GenerationCommand::Commit { generation: 4 },
            GenerationCommand::Revert { generation: 4 },
            GenerationCommand::Purge { generation: 4 },
        ] {
            let encoded = rmp_serde::to_vec(&command).expect("a package command encodes");
            let decoded: GenerationCommand = rmp_serde::from_slice(&encoded).expect("and decodes");
            assert_eq!(decoded, command);
        }
    }

    #[test]
    fn a_precopy_survives_the_daemon_wire_with_its_policy_and_arguments() {
        let mut packages = Packages::new();
        packages.insert(
            proto::module::ModuleSlot::phase("rms_norm", proto::module::PhaseKind::Prefill),
            proto::module::PackageRevision {
                backend: "triton".to_owned(),
                revision: 2,
                arguments: "block = 256".parse().expect("the package's arguments"),
            },
        );
        let command = GenerationCommand::Precopy {
            generation: 9,
            packages: Box::new(packages.clone()),
            policy: proto::module::PackageRevision {
                backend: "fcfs".to_owned(),
                revision: 4,
                arguments: "chunked_prefill_size = 4096"
                    .parse()
                    .expect("the policy's arguments"),
            },
        };

        let wire = command.to_command();
        let back = GenerationCommand::from_command(wire.clone(), 3)
            .expect("the wire form states a package command")
            .expect("and it is the generation layer's");
        // A daemon's precopy states no generation (`proto::process::Precopy`
        // reserves the field, §7.3 step 2): the main process numbers the one it
        // prepares, and `Report::Precopied` echoes the number back. What arrives
        // is therefore the command for the generation after the one serving —
        // the fixture's own number never crosses the wire.
        let expected = GenerationCommand::Precopy {
            generation: 4,
            packages: Box::new(packages.clone()),
            policy: proto::module::PackageRevision {
                backend: "fcfs".to_owned(),
                revision: 4,
                arguments: "chunked_prefill_size = 4096"
                    .parse()
                    .expect("the policy's arguments"),
            },
        };
        assert_eq!(back, expected);

        // The daemon's form carries no modules, only names, revisions and
        // arguments (§6.3), and no step: the boundary is the main process's.
        let bytes = prost::Message::encoded_len(&wire);
        assert!(bytes < 512, "a precopy of one kernel is {bytes} bytes");
        assert_eq!(
            expected.to_command(),
            GenerationCommand::from_command(wire, 3)
                .expect("decodes")
                .expect("is a package command")
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
            let command =
                GenerationCommand::from_command(proto::process::Command { kind: Some(kind) }, 7)
                    .expect("the wire form states a package command")
                    .expect("and it is the generation layer's");
            assert_eq!(command.generation(), 7);
        }
    }

    #[test]
    fn a_command_outside_the_package_layer_is_none() {
        use proto::process::command::Kind;

        for kind in [
            Kind::Park(proto::process::Park {}),
            Kind::Promote(proto::process::Promote::default()),
            Kind::Drain(proto::process::Drain { epoch: 2 }),
            Kind::Pause(proto::process::Pause { paused: true }),
        ] {
            let command =
                GenerationCommand::from_command(proto::process::Command { kind: Some(kind) }, 1)
                    .expect("the command is well formed");
            assert!(command.is_none());
        }
    }
}
