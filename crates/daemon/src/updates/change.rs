//! A configuration change (`PUT /admin/config`): the request, what applying
//! it did, and how in-flight requests continue.
//!
//! `http/admin.rs` builds an [`ApplyRequest`] and renders an [`Applied`].
//! [`resolve`] is the one place a request becomes a target configuration, so
//! the target a dry run reports is the target an apply moves to.

use serde::{Deserialize, Serialize};

use crate::leader::UpdateError;
use crate::options::ServerArgs;

/// How the requests in flight at an epoch's switch continue when the new
/// epoch computes other numerics than the one they ran on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InFlight {
    /// Each continues with the state it reached, which the new epoch reads
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
    /// Report what the change would do and change nothing.
    pub dry_run: bool,
    /// Apply a configuration a rollback marked unavailable.
    pub force: bool,
    /// Seconds from the apply by which the update must be committed: one that
    /// is not is rolled back by the daemon. Unset leaves the trial to the
    /// submitter's and the operator's decision.
    pub deadline_s: Option<f64>,
}

/// What an apply did.
#[derive(Debug, Clone, PartialEq)]
pub enum Applied {
    /// The target is the current configuration.
    Unchanged,
    /// The workers took the options that changed at their next step, or the
    /// daemon reads them from its next request on.
    InPlace(serde_json::Value),
    /// What a dry run found the change would do; nothing changed.
    DryRun(serde_json::Value),
    /// The daemon's own options or executable change: a hot restart of the
    /// daemon with the target configuration applies them. When the change
    /// also moves modules or the build configuration, the daemon's successor
    /// applies `then`, the same change: the daemon changes before what it
    /// updates.
    Daemon {
        config: Box<ServerArgs>,
        executable: Option<String>,
        then: Option<Box<ApplyRequest>>,
    },
    /// An update started, in its generation or its epoch form: the report is
    /// the update as it stood when its `Preparing` began.
    Update(serde_json::Value),
}

/// A module set by operator and revision, as a report names it.
pub(crate) fn modules_report(modules: &crate::epoch::ModuleSet) -> serde_json::Value {
    let kernels = modules
        .kernels
        .iter()
        .map(|(slot, package)| {
            serde_json::json!({
                "op": slot.op,
                "backend": package.backend,
                "revision": package.revision,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "engine": {
            "backend": modules.engine.backend,
            "revision": modules.engine.revision,
        },
        "kernels": kernels,
        "policy": {
            "backend": modules.policy.backend,
            "revision": modules.policy.revision,
        },
    })
}

/// Resolves `request` against the serving configuration `current`: the target
/// configuration, the fields the request stated and the target's update
/// options, which [`crate::epoch::change`] and the update entry decide on.
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
    if request
        .deadline_s
        .is_some_and(|seconds| !(seconds.is_finite() && seconds > 0.0))
    {
        return Err(UpdateError::Refused(
            "the deadline is a positive number of seconds".into(),
        ));
    }
    if target.served_model_name() != current.served_model_name() {
        return Err(UpdateError::Refused(
            "the served model name names the deployment: deploy another one instead".into(),
        ));
    }
    Ok((target, stated, update))
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

    /// `apply` reads the submitter's request: the target configuration, the
    /// overrides, how in-flight requests continue, a dry run, `force` and the
    /// deadline.
    #[test]
    fn an_apply_request_reads_the_operators_body() {
        let request: ApplyRequest = serde_json::from_value(serde_json::json!({
            "replica": 2,
            "config": "[scheduler]\n",
            "set": ["worker_config.executable=/opt/b"],
            "in_flight": "recompute",
            "dry_run": true,
            "force": true,
            "deadline_s": 600.0
        }))
        .expect("the request's shape");
        assert_eq!(request.replica, Some(2));
        assert_eq!(request.config.as_deref(), Some("[scheduler]\n"));
        assert_eq!(request.set, vec!["worker_config.executable=/opt/b"]);
        assert_eq!(request.in_flight, InFlight::Recompute);
        assert!(request.dry_run);
        assert!(request.force);
        assert_eq!(request.deadline_s, Some(600.0));
    }
}
