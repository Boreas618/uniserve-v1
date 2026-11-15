//! An update as the admin API reports it: its phase, its form, what its trial read and why it
//! did not stand, and the update options it reports with it.

use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::config::values;
use crate::process::FaultKind;

/// Where an update is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// The new is made ready beside the old; serving is untouched.
    Preparing,
    /// Service moves at a boundary: to the new, or back to the kept side.
    Switching,
    /// The new serves and the old is kept.
    Trial,
    /// The new serves and the kept side is released.
    Committed,
    /// The kept side serves again and the new is released.
    RolledBack,
    /// The new was never ready: nothing changed.
    Abandoned,
}

impl Phase {
    /// The phase as the status and the event log name it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Preparing => "preparing",
            Self::Switching => "switching",
            Self::Trial => "trial",
            Self::Committed => "committed",
            Self::RolledBack => "rolled_back",
            Self::Abandoned => "abandoned",
        }
    }

    /// Whether the update ended: nothing more moves it.
    pub fn ended(self) -> bool {
        matches!(self, Self::Committed | Self::RolledBack | Self::Abandoned)
    }
}

impl std::fmt::Display for Phase {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The form of an update, as its report names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeForm {
    /// New modules inside the serving engines.
    Generation,
    /// A new epoch beside the serving one.
    Epoch,
}

/// The engines an epoch update moves service between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineCounts {
    /// The serving epoch's engines.
    pub serving: usize,
    /// The target epoch's engines.
    pub target: usize,
}

/// What the previous epoch keeps during its successor's trial
/// (`update_config.previous_epoch`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreviousEpoch {
    /// Its processes, communicators, graph storage, weights and the KV cache
    /// its successor maps: a rollback resumes it at once.
    Parked,
    /// Its weights (and the KV cache its successor maps): a rollback starts
    /// its processes again, which bind them.
    Weights,
    /// The KV cache its successor maps, and in host storage the weights its
    /// successor does not map: a rollback copies them back and starts its
    /// processes again.
    Host,
}

/// What an epoch update reports beyond what every update does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochSwitchReport {
    /// The engine build the target epoch runs.
    pub build: String,
    pub engines: EngineCounts,
    /// The seconds the switch lets the serving engines' requests run on.
    pub drain_s: f64,
    /// Whether the serving epoch's layout differs, so its device memory moves to the target's.
    pub shrink: bool,
    /// What the previous epoch keeps during the trial.
    pub previous_epoch_kept: PreviousEpoch,
    /// The hosts that refused to move the previous epoch's weights to host storage.
    pub host_refused: Vec<String>,
    /// The bytes of the previous epoch's weights moved to host storage.
    pub moved_to_host_bytes: u64,
}

/// An update as the operator reads it: `GET /admin/config`'s `update`, an apply's, commit's
/// or rollback's answer, and the data of the event log's update events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateReport {
    pub change: ChangeForm,
    /// The update's number: the admin operation that applied it.
    pub update: u64,
    pub phase: Phase,
    /// The epoch the update's side executes in.
    pub epoch: u64,
    /// Whether the update is rolling back or rolled back.
    pub rollback: bool,
    /// How long the switch took, once it ended.
    pub switch_ms: Option<u128>,
    /// How long the update has been on trial.
    pub trial_s: Option<f64>,
    /// The seconds left before the deadline rolls the update back.
    pub deadline_s: Option<f64>,
    /// Why the update did not stand, when it did not.
    pub failure: Option<Failure>,
    /// The engines' state the switch could not compare with the target's.
    pub uncompared: Vec<String>,
    /// Why the storage claim that ended the update failed: the storage it released stays
    /// retained until the deployment's next claim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_failed: Option<String>,
    /// The generation the update's engines execute.
    pub generation: Option<u64>,
    /// What an epoch update reports beyond this.
    #[serde(flatten)]
    pub epoch_form: Option<EpochSwitchReport>,
    /// What the trial's thresholds read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trial: Option<TrialReport>,
    /// The serving epoch's steps against their baseline while the target prepares.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfers: Option<StepTime>,
}

/// The thresholds past which a trial rolls back by itself
/// (`update_config.rollback`). None has a default: each is measured on the
/// target workload before it is set.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RollbackConfig {
    /// The share of the requests that end during the trial, in `(0, 1]`,
    /// past which it rolls back when they failed.
    pub failure_rate: Option<f64>,
    /// How much longer than the previous epoch's before its promotion the
    /// new epoch's average step may take, as a fraction (`0.2` for 20%).
    pub step_time_regression: Option<f64>,
    /// How much lower than the previous epoch's before its promotion the
    /// new epoch's token throughput may fall, as a fraction.
    pub throughput_regression: Option<f64>,
    /// The span the step time and throughput are averaged over, before the
    /// promotion and during the trial; required with either regression.
    #[serde(with = "values::optional_duration")]
    pub window: Option<Duration>,
}

/// Average step time and throughput of one epoch over a span.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rates {
    /// Microseconds per batch.
    pub step_us: f64,
    /// Tokens per second the forwards computed over the span the reports cover; `None` when
    /// every report of the span is its scheduler's first, whose span is unknown.
    pub tokens_per_second: Option<f64>,
}

/// What a trial's thresholds read, as the update's report states it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialReport {
    /// The thresholds the trial is read against.
    pub thresholds: RollbackConfig,
    /// How long the trial has run.
    pub seconds: f64,
    /// The requests that ended on trial, and those of them that failed.
    pub requests_ended: u64,
    pub requests_failed: u64,
    /// Each replica's baseline and its rates at the latest reading.
    pub replicas: Vec<ReplicaReading>,
}

/// One replica's side of a trial's reading.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReplicaReading {
    /// The epoch the update serves the replica from.
    pub epoch: u64,
    /// What serving reported before the switch.
    pub baseline: Option<Rates>,
    /// The replica's rates over the window up to the latest reading.
    pub rates: Option<Rates>,
}

/// The throttles one preparation sent, which its change reports.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Throttled {
    pub halvings: u32,
    pub restorations: u32,
}

/// The serving epoch's steps against their baseline while a successor
/// prepares, as the daemon's control last read them, which `GET /admin/config`
/// reports of the change in progress.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StepTime {
    /// The largest ratio of a scheduler's step time to its baseline of the
    /// same batch size among the reports of the latest decision that had a
    /// baseline; `None` before any had.
    pub ratio: Option<f64>,
    /// The slowdown tolerated (`update_config.transfer_slowdown`): a step
    /// may take `1 + slowdown` times its baseline.
    pub slowdown: f64,
    /// Whether the transfers' link budgets stand halved.
    pub halved: bool,
    #[serde(flatten)]
    pub throttled: Throttled,
}

/// A configuration a rollback marked unavailable, and why: a target that was
/// rolled back is not tried again until it is applied with `force`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unavailable {
    /// The digest of what the configuration moves the deployment to: its
    /// build configuration and its module set, revisions included. A revision
    /// installed later is another target.
    pub digest: String,
    /// The epoch that executed it and was rolled back.
    pub epoch: u64,
    pub reason: String,
    /// Seconds since the Unix epoch.
    pub at: f64,
}

/// A threshold of a trial (`update_config.rollback`) that its reading passed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Threshold {
    /// The share of the trial's requests that failed (`failure_rate`).
    FailureRate,
    /// The new epoch's average step against the previous one's
    /// (`step_time_regression`).
    StepTimeRegression,
    /// The new epoch's token throughput against the previous one's
    /// (`throughput_regression`).
    ThroughputRegression,
}

impl Threshold {
    /// The threshold's option name under `update_config.rollback`.
    pub fn name(self) -> &'static str {
        match self {
            Self::FailureRate => "failure_rate",
            Self::StepTimeRegression => "step_time_regression",
            Self::ThroughputRegression => "throughput_regression",
        }
    }
}

/// What made an engine fail or an update not stand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    /// A fault an engine reported, of the kind it stated.
    Fault(FaultKind),
    /// A job of a batch ran past the engine's watchdog.
    Stalled,
    /// A request's logits were not finite.
    NonFiniteLogits,
    /// A scheduling policy failed and its engine went back to the previous policy.
    PolicyRolledBack,
    /// A main process exited, as its daemon observed it.
    ProcessExit,
    /// An engine could not prepare the update's modules.
    PrepareFailed,
    /// An engine rolled the update back by itself.
    RolledBack,
    /// A target process could not be spawned.
    SpawnFailed,
    /// A target's storage could not be claimed.
    ClaimFailed,
    /// An engine runs another build than its epoch's.
    BuildMismatch,
    /// A switched engine executes modules other than the update prepared.
    StaleModules,
    /// A threshold of the trial passed.
    Threshold(Threshold),
    /// The trial was not committed by its deadline.
    Deadline,
    /// The submitter's or the operator's decision.
    Decision,
}

impl FailureKind {
    /// Every kind, each of which has a name of its own: what reading a kind by its name
    /// searches.
    const ALL: [Self; 29] = [
        Self::Fault(FaultKind::Device),
        Self::Fault(FaultKind::Weights),
        Self::Fault(FaultKind::Policy),
        Self::Fault(FaultKind::ProcessExit),
        Self::Fault(FaultKind::Scheduler),
        Self::Fault(FaultKind::Operations),
        Self::Fault(FaultKind::Work),
        Self::Fault(FaultKind::Start),
        Self::Fault(FaultKind::Promote),
        Self::Fault(FaultKind::Commit),
        Self::Fault(FaultKind::Rollback),
        Self::Fault(FaultKind::Drain),
        Self::Fault(FaultKind::Release),
        Self::Fault(FaultKind::Unspecified),
        Self::Stalled,
        Self::NonFiniteLogits,
        Self::PolicyRolledBack,
        Self::ProcessExit,
        Self::PrepareFailed,
        Self::RolledBack,
        Self::SpawnFailed,
        Self::ClaimFailed,
        Self::BuildMismatch,
        Self::StaleModules,
        Self::Threshold(Threshold::FailureRate),
        Self::Threshold(Threshold::StepTimeRegression),
        Self::Threshold(Threshold::ThroughputRegression),
        Self::Deadline,
        Self::Decision,
    ];

    /// The stable short name the admin API and the event log state: one per kind, so a
    /// reader reads back the kind the daemon stated. An engine's own report of a process's
    /// exit is `process_exit_fault`, its daemon's observation of one `process_exit`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Fault(fault) => match fault {
                FaultKind::Device => "device_fault",
                FaultKind::Weights => "weights_fault",
                FaultKind::Policy => "policy_fault",
                FaultKind::ProcessExit => "process_exit_fault",
                FaultKind::Scheduler => "scheduler_fault",
                FaultKind::Operations => "operations_fault",
                FaultKind::Work => "work_fault",
                FaultKind::Start => "start_fault",
                FaultKind::Promote => "promote_fault",
                FaultKind::Commit => "commit_fault",
                FaultKind::Rollback => "rollback_fault",
                FaultKind::Drain => "drain_fault",
                FaultKind::Release => "release_fault",
                FaultKind::Unspecified => "fault",
            },
            Self::Stalled => "stalled",
            Self::NonFiniteLogits => "non_finite_logits",
            Self::PolicyRolledBack => "policy_rolled_back",
            Self::ProcessExit => "process_exit",
            Self::PrepareFailed => "prepare_failed",
            Self::RolledBack => "rolled_back",
            Self::SpawnFailed => "spawn_failed",
            Self::ClaimFailed => "claim_failed",
            Self::BuildMismatch => "build_mismatch",
            Self::StaleModules => "stale_modules",
            Self::Threshold(threshold) => threshold.name(),
            Self::Deadline => "deadline",
            Self::Decision => "decision",
        }
    }

    /// The kind named `name` ([`FailureKind::name`]).
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    /// Whether the failure leaves the failed process's failure domain unusable, so a standby of
    /// it is promoted in its place: a device whose context holds a sticky error, a process of the
    /// engine that exited, or a job that stalled. The other failures fail the operation or the
    /// update they were observed in, and the processes go on.
    pub fn replaces_domain(self) -> bool {
        matches!(
            self,
            Self::Fault(FaultKind::Device | FaultKind::ProcessExit)
                | Self::ProcessExit
                | Self::Stalled
        )
    }
}

impl std::fmt::Display for FailureKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Serialize for FailureKind {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for FailureKind {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::named(&name)
            .ok_or_else(|| serde::de::Error::custom(format!("no failure kind is named `{name}`")))
    }
}

/// Why an engine failed or an update did not stand: a fault an engine observed, a stalled job,
/// non-finite logits, a scheduling policy the engine had to roll back, a failed step of the
/// update, a threshold, the deadline or the operator's decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Failure {
    /// What failed.
    pub kind: FailureKind,
    /// The account given with it, for the daemon's record.
    pub reason: String,
    /// The ordinal of the device the failure happened on, on the host of the process that
    /// reported it, when it names one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<u32>,
}

impl Failure {
    /// A failure of `kind` for `reason`, naming no device.
    pub fn new(kind: FailureKind, reason: String) -> Self {
        Self {
            kind,
            reason,
            device: None,
        }
    }

    /// The failure an engine's report states, or `None` for a report that
    /// states none: an acknowledgement, a lifecycle state, or a failed
    /// preparation, which drops the new generation and leaves the serving
    /// one alone.
    pub fn of_report(report: &crate::process::Report) -> Option<Self> {
        use crate::process::report::Kind;
        let kind = report.kind.as_ref()?;
        match kind {
            Kind::Fault(fault) => Some(Self {
                kind: FailureKind::Fault(fault.kind()),
                reason: fault.reason.clone(),
                device: fault.device,
            }),
            Kind::Stalled(stalled) => Some(Self::new(
                FailureKind::Stalled,
                format!(
                    "the {} job of batch {} ran {} ms",
                    stalled.job, stalled.batch, stalled.elapsed_ms
                ),
            )),
            Kind::NonFiniteLogits(logits) => Some(Self::new(
                FailureKind::NonFiniteLogits,
                format!("non-finite logits for request {}", logits.request),
            )),
            Kind::PolicyRolledBack(policy) => {
                // Absent is the route instance the engine's control
                // thread runs, a value one scheduler thread's policy.
                let instance = match policy.scheduler {
                    Some(scheduler) => format!("scheduler {scheduler}"),
                    None => "the engine's control thread".to_owned(),
                };
                let fatal = if policy.fatal {
                    ", and no previous policy module is left"
                } else {
                    ""
                };
                Some(Self::new(
                    FailureKind::PolicyRolledBack,
                    format!(
                        "the policy of {instance} (backend {} revision {}) failed: {}{fatal}",
                        policy.backend, policy.revision, policy.reason
                    ),
                ))
            }
            Kind::Parked(_)
            | Kind::Serving(_)
            | Kind::Prepared(_)
            | Kind::Switched(_)
            | Kind::Warming(_)
            | Kind::Committed(_)
            | Kind::RolledBack(_)
            | Kind::Drained(_)
            | Kind::Shrunk(_)
            | Kind::PrepareFailed(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every failure kind has a name of its own, which a reader reads back as that kind.
    #[test]
    fn every_failure_kind_reads_back_by_its_name() {
        let mut names = std::collections::BTreeSet::new();
        for kind in FailureKind::ALL {
            let encoded = serde_json::to_value(kind).unwrap();
            assert_eq!(
                serde_json::from_value::<FailureKind>(encoded).unwrap(),
                kind
            );
            names.insert(kind.name());
        }
        assert_eq!(names.len(), FailureKind::ALL.len());
        assert!(serde_json::from_value::<FailureKind>("no_such_kind".into()).is_err());
    }

    /// The reports that fail an update become its failure, with the kind
    /// the engine named; the acknowledgements do not.
    #[test]
    fn the_engines_reports_become_the_updates_failure() {
        use crate::process::report::Kind;

        let fault = crate::process::Report {
            kind: Some(Kind::Fault(crate::process::Fault {
                kind: crate::process::FaultKind::Device.into(),
                reason: "an illegal memory access was encountered".to_owned(),
                device: Some(2),
            })),
        };
        assert_eq!(
            Failure::of_report(&fault),
            Some(Failure {
                kind: FailureKind::Fault(crate::process::FaultKind::Device),
                reason: "an illegal memory access was encountered".to_owned(),
                device: Some(2),
            })
        );

        let stalled = crate::process::Report {
            kind: Some(Kind::Stalled(crate::process::Stalled {
                generation: 4,
                batch: "batch-9".to_owned(),
                job: "copy".to_owned(),
                elapsed_ms: 400,
            })),
        };
        assert_eq!(
            Failure::of_report(&stalled).map(|failure| failure.kind),
            Some(FailureKind::Stalled)
        );

        // The policy's failure names the instance, the backend and the
        // revision, and whether no previous module is left.
        let rolled_back = crate::process::Report {
            kind: Some(Kind::PolicyRolledBack(crate::process::PolicyRolledBack {
                scheduler: Some(2),
                backend: "attention-c".to_owned(),
                revision: 7,
                reason: "the fuel budget is exhausted".to_owned(),
                fatal: false,
            })),
        };
        let failure = Failure::of_report(&rolled_back).expect("a policy rollback fails the update");
        assert_eq!(failure.kind, FailureKind::PolicyRolledBack);
        assert!(failure.reason.contains("scheduler 2"), "{}", failure.reason);
        assert!(failure.reason.contains("attention-c"), "{}", failure.reason);
        assert!(failure.reason.contains("revision 7"), "{}", failure.reason);
        assert!(
            failure.reason.contains("the fuel budget is exhausted"),
            "{}",
            failure.reason
        );
        let route = crate::process::Report {
            kind: Some(Kind::PolicyRolledBack(crate::process::PolicyRolledBack {
                scheduler: None,
                backend: "fcfs".to_owned(),
                revision: 1,
                reason: "a decision was refused".to_owned(),
                fatal: true,
            })),
        };
        let failure = Failure::of_report(&route).expect("a fatal policy rollback fails the update");
        assert!(
            failure.reason.contains("the engine's control thread"),
            "{}",
            failure.reason
        );
        assert!(
            failure.reason.contains("no previous policy module is left"),
            "{}",
            failure.reason
        );

        for acknowledged in [
            crate::process::Report {
                kind: Some(Kind::Prepared(crate::process::Prepared {
                    generation: 4,
                    uncompared: Vec::new(),
                })),
            },
            crate::process::Report {
                kind: Some(Kind::Switched(crate::process::Switched { generation: 4 })),
            },
            crate::process::Report {
                kind: Some(Kind::Committed(crate::process::Committed {})),
            },
            crate::process::Report {
                kind: Some(Kind::RolledBack(crate::process::RolledBack::default())),
            },
            // A failed preparation drops the new generation and leaves the
            // serving one alone: the update is abandoned by its own report.
            crate::process::Report {
                kind: Some(Kind::PrepareFailed(crate::process::PrepareFailed {
                    reason: "the package has no module for this descriptor".to_owned(),
                })),
            },
        ] {
            assert_eq!(Failure::of_report(&acknowledged), None);
        }
    }
}
