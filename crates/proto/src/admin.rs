//! The admin API's schema (`/admin`): the requests a node's admin routes read and the responses
//! they answer, as the leader daemon serves them and as a client of them (`uniserve-manager`, a
//! harness) reads them.
//!
//! The daemon builds and serves these types and its logic stays in the daemon; a client
//! deserializes them, so the two read one schema whatever builds they are of. JSON is the
//! encoding, and the names below are the wire's:
//!
//! - [`Status`] (`GET /admin/status`): what the node serves, idle, starting, serving or stopping,
//!   with the serving deployment's status, the running and the last operation, and the next
//!   event's sequence number.
//! - [`EventsPage`] (`GET /admin/events?after=N&wait=S`, [`EventsQuery`]): the node's event log,
//!   read by long poll.
//! - [`ConfigStatus`] (`GET /admin/config`) and [`ApplyRequest`] (`PUT /admin/config`), answered
//!   with the operation and its [`ConfigChange`].
//! - Every write route answers with its [`Operation`] and, once it succeeded, its result
//!   ([`OperationResponse`]): an [`UpdateReport`] for a commit or a rollback, [`Restarted`] for a
//!   hot restart, [`Paused`] for a pause.

mod config;
mod events;
mod operation;
mod status;
mod update;

pub use config::{
    ApplyRequest, ConfigChange, ConfigStatus, DryRun, EngineLaunch, EpochState, EpochStatus,
    Executing, ExecutingState, Following, Gpu, GpuShare, GpuUuid, InFlight, InPlaceChange,
    KernelPackage, ManagedProcess, Modules, NodeUsage, ProcessState, Quotas, ReplicaDryRun,
    ReplicaStatus, Switched,
};
pub use events::{Event, EventBody, EventsPage, EventsQuery};
pub use operation::{
    Operation, OperationKind, OperationResponse, OperationState, Paused, Restart, Restarted,
    RollbackRequest, Shutdown,
};
pub use status::{DeploymentStatus, NodeState, ServingState, Status, UpdateStatus};
pub use update::{
    ChangeForm, EngineCounts, EpochSwitchReport, Failure, FailureKind, Phase, PreviousEpoch, Rates,
    ReplicaReading, RollbackConfig, StepTime, Threshold, Throttled, TrialReport, Unavailable,
    UpdateReport,
};
