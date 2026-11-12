//! The commands of a generation update.
//!
//! The daemon receives `PUT /admin/config`, finds that the target differs from what serves only
//! in the kernel and policy packages it selects, and moves every serving engine of the epoch:
//!
//! 1. `Prepare` names the packages; the engine resolves and loads them beside the serving
//!    generation and reports `Prepared`, or `PrepareFailed` with nothing changed;
//! 2. `Switch` makes the prepared generation execute from one batch boundary, with the previous
//!    one kept; the engine reports `Switched`;
//! 3. the trial ends in `Commit` or `Rollback`, and the engine releases the generation that no
//!    longer executes once its last batch has completed.
//!
//! The messages carry a package's names — its backend, its revision and its arguments — and never
//! its content: content comes from the Build Server by `resolve`. `PackageRevision`,
//! `ModuleSlot`, `PhaseKind` and `Packages` are the module vocabulary of `proto::module`; this
//! module gives them their form on the peer channel, exactly as the daemon's Protobuf carries
//! them across builds.

use proto::module::Packages;
use serde::{Deserialize, Serialize};

/// A command of the generation update, as an engine receives it.
///
/// The daemon-to-engine form is the Protobuf of `crates/proto`; this is the
/// same command between the engines of one epoch, which are one build and
/// therefore exchange Rust types.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationCommand {
    /// Make `packages` ready as generation `generation` while serving goes on.
    Prepare {
        /// The generation being prepared: the number the engine gives it.
        generation: u64,
        /// The operators' packages to install.
        packages: Box<Packages>,
        /// The scheduling policy the same update installs. It is not a slot of
        /// `packages`: a policy is one package per epoch, named by its own
        /// backend, and the daemon's epoch record keeps the two apart
        /// (`crates/daemon/src/epoch.rs`).
        policy: proto::module::PackageRevision,
    },
    /// Execute `generation` from the batch boundary whose step is `step`.
    Switch {
        /// The generation to execute.
        generation: u64,
        /// The scheduler step the switch takes effect at, agreed by an expert
        /// group's schedulers so that both ends of an all-to-all switch
        /// together.
        ///
        /// `None` when the command came from the daemon, which does not choose
        /// the boundary: the receiving engine's scheduler names the step at
        /// its next batch boundary. `Some` between the engines of one epoch,
        /// whose schedulers agreed on it.
        step: Option<u64>,
    },
    /// The trial ended in a commit.
    Commit {
        /// The generation committed.
        generation: u64,
    },
    /// Return to the generation that executed before the update.
    Rollback {
        /// The generation left.
        generation: u64,
    },
}

impl GenerationCommand {
    /// The generation this command is about.
    pub const fn generation(&self) -> u64 {
        match self {
            Self::Prepare { generation, .. }
            | Self::Switch { generation, .. }
            | Self::Commit { generation }
            | Self::Rollback { generation } => *generation,
        }
    }

    /// The stage's name, for tracing.
    pub const fn stage(&self) -> &'static str {
        match self {
            Self::Prepare { .. } => "prepare",
            Self::Switch { .. } => "switch",
            Self::Commit { .. } => "commit",
            Self::Rollback { .. } => "rollback",
        }
    }
}

/// These commands cross two wires with one shape: the daemon's Protobuf
/// (`proto::process::Command`) and the peer channel of one epoch's engines,
/// which are one build and exchange this Rust type. The mapping is
/// written once, here, so a field cannot be carried by one wire and dropped by
/// the other without a reader seeing it.
impl GenerationCommand {
    /// This command as the daemon's Protobuf `Command`.
    ///
    /// The switch's `step` is not part of the daemon's form: the boundary is
    /// the engine's own decision.
    pub fn to_command(&self) -> proto::process::Command {
        use proto::process::command::Kind;

        let kind = match self {
            Self::Prepare {
                packages, policy, ..
            } => Kind::Prepare(proto::process::Prepare {
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
            Self::Switch { generation, .. } => Kind::Switch(proto::process::Switch {
                generation: *generation,
            }),
            Self::Commit { .. } => Kind::Commit(proto::process::Commit {}),
            Self::Rollback { .. } => Kind::Rollback(proto::process::Rollback {
                reason: String::new(),
            }),
        };
        proto::process::Command { kind: Some(kind) }
    }

    /// The generation command `command` states, for an engine executing
    /// generation `current`.
    ///
    /// `None` for the daemon's other commands (drain, promote, pause, serve),
    /// which move the engine's lifecycle and not its generation. The daemon's
    /// commit and rollback carry no generation - it is the one executing when
    /// they arrive - so the number is `current` (for a rollback, the generation
    /// being left).
    pub fn from_command(
        command: proto::process::Command,
        current: u64,
    ) -> Result<Option<Self>, crate::IpcError> {
        use proto::process::command::Kind;

        let Some(kind) = command.kind else {
            return Err(command_error("a command states no kind"));
        };
        Ok(Some(match kind {
            Kind::Prepare(prepare) => {
                let mut packages = Packages::new();
                for kernel in prepare.kernels {
                    let package = kernel
                        .package
                        .ok_or_else(|| command_error("a prepare's kernel states no package"))?;
                    let phase = kernel.phase.map(phase_from_wire).transpose()?.flatten();
                    let slot = proto::module::ModuleSlot {
                        op: kernel.op,
                        phase,
                        part: kernel.part,
                    };
                    packages.insert(slot, package_from_wire(package)?);
                }
                let policy = prepare
                    .policy
                    .ok_or_else(|| command_error("a prepare states no policy"))?;
                Self::Prepare {
                    // The daemon's prepare states no number: the engine numbers
                    // the generation it prepares, and `Report::Prepared` states
                    // it.
                    generation: current + 1,
                    packages: Box::new(packages),
                    policy: package_from_wire(policy)?,
                }
            }
            Kind::Switch(switch) => Self::Switch {
                generation: switch.generation,
                step: None,
            },
            Kind::Commit(_) => Self::Commit {
                generation: current,
            },
            Kind::Rollback(_) => Self::Rollback {
                generation: current,
            },
            _ => return Ok(None),
        }))
    }
}

/// A package as the daemon's Protobuf carries it: its backend, its revision and
/// its arguments, never the module's bytes.
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
        let command = GenerationCommand::Switch {
            generation: 4,
            step: Some(17),
        };
        assert_eq!(command.generation(), 4);
        assert_eq!(command.stage(), "switch");
    }

    #[test]
    fn a_prepare_carries_package_names_and_arguments_and_never_content() {
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
        let command = GenerationCommand::Prepare {
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

        let encoded = rmp_serde::to_vec(&command).expect("a prepare encodes");
        // The module's bytes are not part of the command: an engine resolves them from its
        // Build Server, and the peer channel carries names only.
        assert!(
            encoded.len() < 1024,
            "a prepare of two packages is {} bytes",
            encoded.len()
        );
        let decoded: GenerationCommand = rmp_serde::from_slice(&encoded).expect("and decodes");
        assert_eq!(decoded, command);
        let GenerationCommand::Prepare {
            generation,
            packages,
            policy,
        } = decoded
        else {
            panic!("a prepare decoded as {}", decoded.stage());
        };
        assert_eq!(generation, 5);
        assert_eq!(
            packages[&proto::module::ModuleSlot::all("decode_attention")].revision,
            3
        );
        assert_eq!(policy.backend, "fcfs");
    }

    #[test]
    fn the_commands_between_engines_survive_the_peer_encoding() {
        for command in [
            GenerationCommand::Switch {
                generation: 4,
                step: Some(17),
            },
            GenerationCommand::Commit { generation: 4 },
            GenerationCommand::Rollback { generation: 4 },
        ] {
            let encoded = rmp_serde::to_vec(&command).expect("a generation command encodes");
            let decoded: GenerationCommand = rmp_serde::from_slice(&encoded).expect("and decodes");
            assert_eq!(decoded, command);
        }
    }

    #[test]
    fn a_prepare_survives_the_daemon_wire_with_its_policy_and_arguments() {
        let mut packages = Packages::new();
        packages.insert(
            proto::module::ModuleSlot::phase("rms_norm", proto::module::PhaseKind::Prefill),
            proto::module::PackageRevision {
                backend: "triton".to_owned(),
                revision: 2,
                arguments: "block = 256".parse().expect("the package's arguments"),
            },
        );
        let policy = proto::module::PackageRevision {
            backend: "fcfs".to_owned(),
            revision: 4,
            arguments: "chunked_prefill_size = 4096"
                .parse()
                .expect("the policy's arguments"),
        };
        let command = GenerationCommand::Prepare {
            generation: 9,
            packages: Box::new(packages.clone()),
            policy: policy.clone(),
        };

        let wire = command.to_command();
        let back = GenerationCommand::from_command(wire.clone(), 3)
            .expect("the wire form states a generation command")
            .expect("and it is a generation's");
        // A daemon's prepare states no generation: the engine numbers the one it prepares, and
        // `Report::Prepared` states the number. What arrives is therefore the command for the
        // generation after the one that executes; the fixture's own number never crosses the wire.
        let expected = GenerationCommand::Prepare {
            generation: 4,
            packages: Box::new(packages),
            policy,
        };
        assert_eq!(back, expected);

        // The daemon's form carries no modules, only names, revisions and arguments, and no
        // step: the boundary is the engine's.
        let bytes = prost::Message::encoded_len(&wire);
        assert!(bytes < 512, "a prepare of one kernel is {bytes} bytes");
    }

    #[test]
    fn the_daemons_commit_and_rollback_name_the_generation_the_engine_executes() {
        use proto::process::command::Kind;

        for kind in [
            Kind::Commit(proto::process::Commit {}),
            Kind::Rollback(proto::process::Rollback {
                reason: String::new(),
            }),
        ] {
            let command =
                GenerationCommand::from_command(proto::process::Command { kind: Some(kind) }, 7)
                    .expect("the wire form states a generation command")
                    .expect("and it is a generation's");
            assert_eq!(command.generation(), 7);
        }
    }

    #[test]
    fn a_lifecycle_command_is_no_generation_command() {
        use proto::process::command::Kind;

        for kind in [
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
