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

use program_server::Programs;
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
        /// The operators' programs and the scheduling policy to install, and
        /// the programs they replace (kept through probation, §7.3 step 4).
        programs: Box<Programs>,
    },
    /// §7.3 step 3: cut over at the batch boundary whose step is `step`.
    Switchover {
        /// The generation to serve.
        generation: u64,
        /// The scheduler step the switchover runs at, agreed by an expert
        /// group's schedulers so that both ends of an all-to-all switch
        /// together.
        step: u64,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_command_names_its_generation_and_stage() {
        let command = ProgramCommand::Switchover {
            generation: 4,
            step: 17,
        };
        assert_eq!(command.generation(), 4);
        assert_eq!(command.stage(), "switchover");
    }

    #[test]
    fn the_commands_between_main_processes_survive_the_peer_encoding() {
        for command in [
            ProgramCommand::Switchover {
                generation: 4,
                step: 17,
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
}
