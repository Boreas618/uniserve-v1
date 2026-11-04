//! The vocabulary of a program-layer update (`specs/worker.md` §7.3).
//!
//! A program-layer update moves through precopy, switchover, probation and
//! commit or revert, and ends with a purge. The daemon drives it and every main
//! process of the epoch takes part, so the names below travel: the daemon and
//! the main processes of one epoch are one build and exchange them over the ipc
//! channel (§2.3); the daemon's own process interface carries the same names in
//! its append-only Protobuf (`crates/proto`) so that a later build of the daemon
//! keeps serving the main processes of this one.
//!
//! Only the vocabulary lives here. The commands that carry programs are in
//! [`crate::program`], because a `Precopy` names the [`Program`] set to install,
//! and a program is defined by the Program Server (§5.1).

use crate::{RequestKey, Stall};

/// The first error of a failed update, as §7.3 step 4 lists the conditions that
/// end probation in a revert, and §7.3 step 2 the failure that discards a new
/// generation.
///
/// The daemon marks the update unavailable with the first of these until it is
/// applied again by force; a main process only reports its own.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateFailure {
    /// Precopy failed: the new generation was discarded and serving goes on
    /// with the old one (§7.3 step 2). This one does not revert anything,
    /// because nothing was switched over.
    Precopy {
        /// What failed: a module that did not resolve, a shadow batch outside
        /// its tolerance, a graph that did not recapture.
        reason: String,
    },
    /// A device error that leaves its context unusable, such as an illegal
    /// address: the epoch's service cannot go on (§2.5).
    Device {
        /// The device, by ordinal.
        device: u32,
        /// The driver's error.
        error: String,
        /// The batch that was in flight on it, if any.
        batch: Option<u64>,
    },
    /// A job stayed inside one executor past half the watchdog timeout (§2.4).
    Stalled {
        /// The job's batch and how long it has run.
        stall: Stall,
    },
    /// A batch produced non-finite logits (§7.3 step 4).
    NonFiniteLogits {
        /// The request whose logits were not finite.
        request: RequestKey,
    },
    /// The scheduling policy failed: a trap, fuel, a limit outside the build
    /// config, a decision the mechanism rejected, or a starved request (§4.4).
    Policy {
        /// The mechanism's error.
        error: String,
    },
    /// The update's rollback threshold of `update_config.rollback` was
    /// exceeded (§7.3 step 4). The daemon decides this one; a main process
    /// reports what the threshold is measured on.
    Threshold {
        /// What crossed the threshold.
        reason: String,
    },
}

impl UpdateFailure {
    /// Whether this failure reverts an update on probation. A precopy failure
    /// does not: it discards the new generation before any switchover.
    pub const fn reverts(&self) -> bool {
        !matches!(self, Self::Precopy { .. })
    }

    /// The failure as the one-line reason a record or a log keeps.
    pub fn reason(&self) -> String {
        match self {
            Self::Precopy { reason }
            | Self::Policy { error: reason }
            | Self::Threshold { reason } => reason.clone(),
            Self::Device { device, error, .. } => format!("device {device}: {error}"),
            Self::Stalled { stall } => format!(
                "a job of batch {:?} ran {} ms",
                stall.batch_id, stall.running_ms
            ),
            Self::NonFiniteLogits { request } => {
                format!(
                    "request {:?} produced non-finite logits",
                    request.request_id
                )
            }
        }
    }
}

/// What a main process reports of a program-layer update to the daemon and to
/// the scheduler of its epoch (§7.3).
///
/// Every status names the generation it is about. A `Failed` and a
/// `PolicyReverted` are sent unasked, like a stall report; the others answer the
/// command that caused them.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramStatus {
    /// Precopy finished: the new generation is loaded, rebound or recaptured,
    /// and its shadow batches are within tolerance. Serving is unchanged.
    Precopied {
        /// The generation being installed.
        generation: u64,
    },
    /// Precopy failed: the new generation was discarded and serving goes on.
    PrecopyFailed {
        /// The generation that was being installed.
        generation: u64,
        /// The first error.
        failure: UpdateFailure,
    },
    /// The update switched over at a batch boundary; the old generation is
    /// kept for probation.
    Switched {
        /// The generation now serving.
        generation: u64,
    },
    /// Probation ended in a commit; the old generation is purged next.
    Committed {
        /// The generation committed.
        generation: u64,
    },
    /// The update was reverted and the old generation serves again.
    Reverted {
        /// The generation that was reverted away from.
        generation: u64,
    },
    /// The generation was purged: the last batch of the generation before it
    /// completed and its graphs and kernels are released (§7.3 step 6).
    Purged {
        /// The generation purged.
        generation: u64,
    },
    /// An update on probation failed and was reverted automatically; this
    /// carries the first error (§7.3 step 4).
    Failed {
        /// The generation that failed.
        generation: u64,
        /// The first error.
        failure: UpdateFailure,
    },
    /// After a commit, the scheduling policy failed and the mechanism
    /// instantiated the previous policy on the state it holds (§4.4, §7.3).
    /// The epoch keeps serving; the daemon records the previous policy again.
    PolicyReverted {
        /// The generation whose policy failed.
        generation: u64,
        /// The policy's error.
        failure: UpdateFailure,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_precopy_failure_does_not_revert_an_update() {
        let failure = UpdateFailure::Precopy {
            reason: "a module did not resolve".into(),
        };
        assert!(!failure.reverts());
        assert_eq!(failure.reason(), "a module did not resolve");
    }

    #[test]
    fn a_device_failure_reverts_and_names_its_device() {
        let failure = UpdateFailure::Device {
            device: 3,
            error: "an illegal address was encountered".into(),
            batch: Some(41),
        };
        assert!(failure.reverts());
        assert_eq!(
            failure.reason(),
            "device 3: an illegal address was encountered"
        );
    }

    #[test]
    fn a_stall_names_its_batch_and_duration() {
        let failure = UpdateFailure::Stalled {
            stall: Stall {
                batch_id: Some(9),
                route: Some("decode".into()),
                method: None,
                calls: Vec::new(),
                running_ms: 15_000,
            },
        };
        assert_eq!(
            failure.reason(),
            "a job of batch Some(9) ran 15000 ms"
        );
    }
}
