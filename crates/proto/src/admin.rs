//! The schema of what a node's admin API answers beside the policy package: the fields of the
//! observation rows the leader daemon writes, which `/admin/query` reads, as their writers build
//! them and their readers (`uniserve-manager`, a harness) deserialize them.
//!
//! The admin API has two endpoints, `/admin/policy` (the package) and `/admin/query` (SQL over
//! the `stats` and `events` tables). JSON is the encoding of a row's fields, and the names below
//! are the wire's:
//!
//! - [`UpdateReport`]: an update as its `log` events of source `update` state it, one for each
//!   phase it enters.
//! - [`ManagedProcess`]: an engine process as its `process` events state it.
//! - [`ServingState`]: a replica's engine state, as the `uniserve_replica_state` gauge and the
//!   `log` events of source `engine` name it.
//! - [`EpochState`], [`NodeUsage`]: an epoch's state and a node's storage, which the sampled
//!   gauges state.

mod config;
mod state;
mod update;

pub use config::{
    EpochState, EpochStatus, Gpu, GpuUuid, InFlight, KernelPackage, ManagedProcess, Modules,
    NodeUsage, ProcessState,
};
pub use state::ServingState;
pub use update::{
    ChangeForm, EngineCounts, EpochSwitchReport, Failure, FailureKind, Phase, PreviousEpoch, Rates,
    ReplicaReading, RollbackConfig, StepTime, Threshold, Throttled, TrialReport, Unavailable,
    UpdateReport,
};
