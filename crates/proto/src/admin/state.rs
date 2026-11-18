//! The serving state of a replica's engine, as the `uniserve_replica_state` gauge and the `log`
//! events of source `engine` name it.

use serde::{Deserialize, Serialize};

/// The lifecycle state of the engine a replica's requests go to.
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
    /// The state's name, as the gauge's `state` label and the events state it.
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
