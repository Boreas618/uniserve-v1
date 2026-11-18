//! The admin API's write operations: what each is, where it stands, and what the routes whose
//! bodies and results are not a configuration's read and answer.

use serde::{Deserialize, Serialize};

/// A write operation: running, or one that ended. The status states the running one and the
/// last that ended, and the event log every start and end.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Operation {
    /// Its number, from 1, increasing by one across the node's deployments.
    pub id: u64,
    pub kind: OperationKind,
    pub state: OperationState,
    /// Seconds since the Unix epoch.
    pub started: f64,
    /// Seconds since the Unix epoch, once it ended.
    pub ended: Option<f64>,
    /// Why it failed.
    pub error: Option<String>,
}

/// What an operation does: the route that started it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    /// `PUT /admin/config`: a change of the configuration, or an idle node's bring-up of a
    /// deployment.
    Config,
    /// `POST /admin/config/commit`.
    Commit,
    /// `POST /admin/config/rollback`.
    Rollback,
    /// `POST /admin/rebalance_experts`.
    RebalanceExperts,
    /// `POST /admin/pause_generation`.
    PauseGeneration,
    /// `POST /admin/continue_generation`.
    ContinueGeneration,
    /// `POST /admin/flush_cache`.
    FlushCache,
    /// `POST /admin/sleep`.
    Sleep,
    /// `POST /admin/wake_up`.
    WakeUp,
    /// `POST /admin/load_lora_adapter`.
    LoadLoraAdapter,
    /// `POST /admin/unload_lora_adapter`.
    UnloadLoraAdapter,
    /// `POST` or `DELETE /admin/profile`.
    Profile,
    /// `POST /admin/restart`.
    Restart,
    /// `POST /admin/shutdown`.
    Shutdown,
}

impl OperationKind {
    /// The kind's name, as the status and the event log state it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Config => "config",
            Self::Commit => "commit",
            Self::Rollback => "rollback",
            Self::RebalanceExperts => "rebalance_experts",
            Self::PauseGeneration => "pause_generation",
            Self::ContinueGeneration => "continue_generation",
            Self::FlushCache => "flush_cache",
            Self::Sleep => "sleep",
            Self::WakeUp => "wake_up",
            Self::LoadLoraAdapter => "load_lora_adapter",
            Self::UnloadLoraAdapter => "unload_lora_adapter",
            Self::Profile => "profile",
            Self::Restart => "restart",
            Self::Shutdown => "shutdown",
        }
    }
}

impl std::fmt::Display for OperationKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Where an operation stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationState {
    Running,
    Succeeded,
    /// It failed, for the operation's `error`.
    Failed,
}

impl OperationState {
    /// The state's name, as the status and the event log state it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
        }
    }
}

impl std::fmt::Display for OperationState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// What a write route answers: its operation and, once the operation succeeded, its result
/// (`R`, the route's own). A route asked not to wait (`"wait": false`) answers `202` with the
/// operation running and no result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationResponse<R> {
    #[serde(flatten)]
    pub operation: Operation,
    pub result: Option<R>,
}

/// The body of `POST /admin/config/rollback`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RollbackRequest {
    /// Why the update is rolled back, which the unavailable mark records; the daemon states
    /// the decision itself when unset.
    pub reason: Option<String>,
}

/// The body of `POST /admin/shutdown`, which its result repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Shutdown {
    /// Release the deployment's weights and caches at the host's Storage Server once its
    /// engine is down; otherwise they stay for a restart of the same deployment.
    pub retire: bool,
}

impl Default for Shutdown {
    /// A shutdown releases the deployment's storage unless it states otherwise.
    fn default() -> Self {
        Self { retire: true }
    }
}

/// The body of `POST /admin/restart`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Restart {
    /// Answer once the successor serves; otherwise at once (`202`).
    pub wait: bool,
}

impl Default for Restart {
    /// A restart's answer waits for the successor unless it states otherwise.
    fn default() -> Self {
        Self { wait: true }
    }
}

/// The result of a hot restart of the daemon: the successor that serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Restarted {
    /// The successor daemon's process id.
    pub pid: u32,
}

/// The result of `POST /admin/pause_generation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Paused {
    /// Whether new requests already waited.
    pub already_paused: bool,
}
