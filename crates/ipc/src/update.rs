//! The vocabulary of a generation update.
//!
//! A generation update moves a serving engine from the kernel and policy modules it executes to
//! others: it is prepared beside the serving generation, switched to at one batch boundary, served
//! on trial with the previous generation kept, and committed or rolled back. The daemon drives it
//! through its process interface (`crates/proto`), whose reports acknowledge each step; the
//! failures an engine observes meanwhile are named below.
//!
//! Only the vocabulary lives here. The commands that carry packages are in
//! [`crate::generation`], because a `Prepare` names the [`PackageRevision`] set to install,
//! and a package is defined by the Build Server.

use crate::{RequestKey, Stall};

/// The first error of a failed update: what ends a trial in a rollback, or drops a
/// generation that was being prepared.
///
/// The daemon marks the update unavailable with the first of these until it is
/// applied again by force; an engine only reports its own.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateFailure {
    /// Preparing failed: the new generation was discarded and serving goes on
    /// with the one that executes. Nothing is rolled back, because nothing was
    /// switched.
    Prepare {
        /// What failed: a module that did not resolve, a shadow batch outside
        /// its tolerance, a graph that did not recapture.
        reason: String,
    },
    /// A device error that leaves its context unusable, such as an illegal
    /// address: the engine cannot go on serving.
    Device {
        /// The device, by ordinal.
        device: u32,
        /// The driver's error.
        error: String,
        /// The batch that was in flight on it, if any.
        batch: Option<u64>,
    },
    /// A job stayed inside one worker past half the watchdog timeout.
    Stalled {
        /// The job's batch and how long it has run.
        stall: Stall,
    },
    /// A batch produced non-finite logits.
    NonFiniteLogits {
        /// The request whose logits were not finite.
        request: RequestKey,
    },
    /// The scheduling policy failed: a trap, fuel, a limit outside the build
    /// config, a decision the mechanism rejected, or a starved request.
    Policy {
        /// The mechanism's error.
        error: String,
    },
    /// The daemon ended the trial: a threshold of `update_config.rollback`, the
    /// deadline of the apply, or the operator's decision. The daemon decides
    /// this one; an engine reports what the thresholds are measured on.
    Threshold {
        /// What crossed the threshold.
        reason: String,
    },
}

impl UpdateFailure {
    /// Whether this failure rolls back an update on trial. A failure of the
    /// preparation does not: it discards the new generation before any switch.
    pub const fn rolls_back(&self) -> bool {
        !matches!(self, Self::Prepare { .. })
    }

    /// The failure as the one-line reason a record or a log keeps.
    pub fn reason(&self) -> String {
        match self {
            Self::Prepare { reason }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_preparation_rolls_back_nothing() {
        let failure = UpdateFailure::Prepare {
            reason: "a module did not resolve".into(),
        };
        assert!(!failure.rolls_back());
        assert_eq!(failure.reason(), "a module did not resolve");
    }

    #[test]
    fn a_device_failure_rolls_back_and_names_its_device() {
        let failure = UpdateFailure::Device {
            device: 3,
            error: "an illegal address was encountered".into(),
            batch: Some(41),
        };
        assert!(failure.rolls_back());
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
        assert_eq!(failure.reason(), "a job of batch Some(9) ran 15000 ms");
    }
}
