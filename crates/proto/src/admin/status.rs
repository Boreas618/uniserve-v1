//! What a node states of itself (`GET /admin/status`).

use serde::{Deserialize, Serialize};

use super::operation::Operation;
use super::update::Phase;

/// `GET /admin/status`: what the node serves, the running and the last operation, and the
/// sequence number of the next event of its log. A node answers it whatever it serves, so a
/// client reads one route for an idle, a starting, a serving and a stopping node alike.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Status {
    /// The daemon's version (`CARGO_PKG_VERSION`).
    pub version: String,
    /// What the node serves (`state`, and what each state carries).
    #[serde(flatten)]
    pub node: NodeState,
    /// The operation that runs.
    pub operation: Option<Operation>,
    /// The operation that ended last.
    pub last_operation: Option<Operation>,
    /// The sequence number the next event of the node's log takes: a reader that starts from
    /// now reads the events after `next_event - 1`.
    pub next_event: u64,
}

/// What a node serves, as its status states it (`state`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum NodeState {
    /// No deployment: a `PUT /admin/config` brings one up. `last_error` is why the last
    /// deployment failed to come up or to come down.
    Idle { last_error: Option<String> },
    /// The deployment `name` comes up.
    Starting { name: String },
    /// A deployment serves: its status, and the daemon's readiness record once it reported
    /// serving.
    Serving {
        deployment: Box<DeploymentStatus>,
        readiness: Option<Box<service::Readiness>>,
    },
    /// The deployment `name` comes down.
    Stopping { name: String },
}

/// What a serving deployment serves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeploymentStatus {
    /// The served model name, the deployment's name.
    pub deployment: String,
    /// The model the deployment serves.
    pub model: String,
    /// The lifecycle state of the engine requests go to: `serving` while any
    /// replica's engine serves, otherwise the first replica's.
    pub engine: ServingState,
    /// Why the engine failed or no engine will serve.
    pub reason: Option<String>,
    /// Whether new requests wait (`/admin/pause_generation`).
    pub paused: bool,
    /// Whether the engine gave back its KV cache and graph storage
    /// (`/admin/sleep`).
    pub asleep: bool,
    /// Whether an engine is being replaced: by an update or a relaunch.
    pub replacing: bool,
    /// The requests the serving engine has in flight.
    pub in_flight: usize,
    /// The new requests the daemon holds before any engine takes them: while
    /// generation is paused, the engine sleeps, or no engine is ready.
    pub held: usize,
    /// The launch generation of the first replica's engine (the scheduler
    /// node's, for a worker group over several hosts); `None` while none
    /// serves, or for an engine this daemon did not start. It names that
    /// process's row in `GET /admin/config`'s `processes`.
    pub generation: Option<u64>,
    /// The launch generation of each engine of the parked standby beside
    /// the first replica's engine (`worker_config.standby`), one per host in
    /// the world's order: the serving `generation` a promotion leaves is its
    /// first. Empty while there is none.
    pub standby: Vec<u64>,
    /// The update in flight; `GET /admin/config` holds its whole report.
    pub update: Option<UpdateStatus>,
}

/// The lifecycle state of the engine a deployment's requests go to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServingState {
    /// It loads its ranks, resolves its modules and captures its graphs.
    Starting,
    /// It executes and admits nothing.
    Parked,
    /// It admits and executes requests.
    Serving,
    /// It admits nothing more; its requests end as migratable.
    Draining,
    /// It shut down, or its connection closed.
    Stopped,
    /// It serves nothing more: its start failed or its ranks are lost.
    Failed,
    /// An engine is being replaced, and requests wait for the next.
    Holding,
    /// The successor daemon of a hot restart serves the requests.
    HandedOver,
    /// No engine will serve.
    Closed,
}

impl ServingState {
    /// The state's name, as the status serializes it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Parked => "parked",
            Self::Serving => "serving",
            Self::Draining => "draining",
            Self::Stopped => "stopped",
            Self::Failed => "failed",
            Self::Holding => "holding",
            Self::HandedOver => "handed_over",
            Self::Closed => "closed",
        }
    }
}

impl std::fmt::Display for ServingState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The update in flight, as the deployment's status names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateStatus {
    /// The admin operation that applied it.
    pub number: u64,
    /// `preparing`, `switching` or `trial`.
    pub phase: Phase,
}
