//! A configuration change (`PUT /admin/config`): the request's shape, what
//! applying it changed, and how in-flight requests continue.
//!
//! `http/admin.rs` builds an [`ApplyRequest`] and lowers an [`Applied`]; the
//! epoch layer reads both before it starts a switch. [`resolve`] is the one
//! place a request becomes a target configuration, so the target an operator
//! sees planned is the target the leader applies. The types keep the names
//! and wire shapes the API already uses.

use serde::{Deserialize, Serialize};

use crate::leader::UpdateError;
use crate::options::ServerArgs;

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
    /// also changes what the main processes are built from, the daemon's
    /// successor applies `then`, the same change, which the epoch layer then
    /// switches: the orchestrator changes before what it orchestrates.
    Daemon {
        config: Box<ServerArgs>,
        executable: Option<String>,
        then: Option<Box<ApplyRequest>>,
    },
    /// The epoch layer's switch runs on trial: the target's main processes
    /// serve while the previous epoch is parked (§7.4).
    Epoch(serde_json::Value),
    /// The program layer's update runs on trial inside the serving epoch: its
    /// processes switched to the target's programs and are on probation
    /// (§7.2, §7.3).
    Program(serde_json::Value),
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

/// Resolves `request` against the serving configuration `current`: the target
/// configuration, the fields the request stated and the target's update
/// options, which [`crate::epoch::layer`] and the update entry decide on. The
/// admin API resolves a change here before it routes it, so the target an
/// operator sees planned is the target the leader applies.
pub(crate) fn resolve(
    current: &ServerArgs,
    request: &ApplyRequest,
) -> Result<
    (
        ServerArgs,
        crate::options::Stated,
        crate::options::UpdateConfig,
    ),
    UpdateError,
> {
    let (mut target, stated) = match &request.config {
        Some(text) => crate::options::resolve_text(text, &request.set),
        None => crate::options::overridden(current, &request.set)
            .map(|(config, paths)| (config, crate::options::Stated::from_paths(paths))),
    }
    .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
    target
        .validate(&stated)
        .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
    let update = target.update_config.clone();
    if request.config.is_none() {
        target.update_config = current.update_config.clone();
    }
    if target.served_model_name() != current.served_model_name() {
        return Err(UpdateError::Refused(
            "the served model name names the deployment: deploy another one instead".into(),
        ));
    }
    Ok((target, stated, update))
}
