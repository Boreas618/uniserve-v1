//! A configuration change (`PUT /admin/config`): the request's shape, what
//! applying it changed, and how in-flight requests continue.
//!
//! `http/admin.rs` builds an [`ApplyRequest`] and lowers an [`Applied`]; the
//! epoch layer reads both before it starts a switch. The types keep the names
//! and wire shapes the API already uses.

use serde::{Deserialize, Serialize};

use crate::options::ServeConfig;

/// How the requests in flight at a promotion continue when the new world
/// computes other numerics than the one they ran on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InFlight {
    /// Each continues with the state it reached, which the new world reads
    /// as its own; the pages it writes afterwards reach no shared tier.
    #[default]
    Keep,
    /// Each computes its state again from its tokens, exactly.
    Recompute,
}

/// A change of the deployment's configuration (`PUT /admin/config`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ApplyRequest {
    /// Update this independent replica. Without an index, a deployment
    /// update advances through replicas as each trial is committed.
    pub replica: Option<usize>,
    /// The whole target configuration as TOML, as `--config` reads it;
    /// `None` takes the current one.
    pub config: Option<String>,
    /// `PATH=VALUE` assignments over that configuration, as `--set` states
    /// them.
    pub set: Vec<String>,
    pub in_flight: InFlight,
    /// Report the plan and change nothing.
    pub dry_run: bool,
    /// Apply a configuration a rollback marked unavailable.
    pub force: bool,
}

/// What an apply changed.
#[derive(Debug, Clone, PartialEq)]
pub enum Applied {
    /// The target is the current configuration.
    Unchanged,
    /// The schedulers took the new limits at their next step, or the daemon
    /// reads the options that changed from its next request on.
    InPlace(serde_json::Value),
    /// The plan of a dry run, which changed nothing.
    Planned(serde_json::Value),
    /// The daemon's own options or executable change: a hot restart of the
    /// daemon with the target configuration applies them. When the change
    /// also changes what a world executes, the daemon's successor applies
    /// `then`, the same change, which then changes the world alone: the
    /// orchestrator changes before what it orchestrates.
    Daemon {
        config: Box<ServeConfig>,
        executable: Option<String>,
        then: Option<Box<ApplyRequest>>,
    },
    /// A new world executes on trial.
    World(serde_json::Value),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The API's spelling of the two continuation modes is the wire the
    /// operator sends; `keep` is the default when the field is absent.
    #[test]
    fn in_flight_is_snake_case_and_defaults_to_keep() {
        assert_eq!(serde_json::to_value(InFlight::Keep).unwrap(), "keep");
        assert_eq!(
            serde_json::to_value(InFlight::Recompute).unwrap(),
            "recompute"
        );
        assert_eq!(
            serde_json::from_str::<InFlight>("\"recompute\"").unwrap(),
            InFlight::Recompute
        );
        assert_eq!(InFlight::default(), InFlight::Keep);
    }

    /// `apply` reads the operator's request: the target configuration, the
    /// overrides, how in-flight requests continue, a dry run and `force`.
    #[test]
    fn an_apply_request_reads_the_operators_body() {
        let request: ApplyRequest = serde_json::from_value(serde_json::json!({
            "replica": 2,
            "config": "[scheduler]\n",
            "set": ["worker_config.executable=/opt/b"],
            "in_flight": "recompute",
            "dry_run": true,
            "force": true
        }))
        .expect("the request's shape");
        assert_eq!(request.replica, Some(2));
        assert_eq!(request.config.as_deref(), Some("[scheduler]\n"));
        assert_eq!(request.set, vec!["worker_config.executable=/opt/b"]);
        assert_eq!(request.in_flight, InFlight::Recompute);
        assert!(request.dry_run);
        assert!(request.force);
    }
}
