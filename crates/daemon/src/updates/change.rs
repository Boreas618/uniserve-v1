//! A configuration change (`PUT /admin/config`): the request, what applying
//! it did, and how in-flight requests continue.
//!
//! `http/admin.rs` reads an [`ApplyRequest`] and renders an [`Applied`].
//! [`resolve`] is the one place a request becomes a target configuration, so
//! the target a dry run reports is the target an apply moves to.

use proto::admin::{ApplyRequest, DryRun, InPlaceChange, Modules, UpdateReport};

use crate::leader::UpdateError;
use crate::options::ServerArgs;

/// What an apply did.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Applied {
    /// The target is the current configuration.
    Unchanged,
    /// The replicas took the options that changed at their engines' next step, or the
    /// daemon reads them from its next request on.
    InPlace(Vec<InPlaceChange>),
    /// What a dry run found the change would do; nothing changed.
    DryRun(DryRun),
    /// The daemon's own options change: a hot restart of the
    /// daemon with the target configuration applies them. When the change
    /// also moves modules or the build configuration, the daemon's successor
    /// applies `then`, the same change: the daemon changes before what it
    /// updates.
    Daemon {
        config: Box<ServerArgs>,
        then: Option<Box<ApplyRequest>>,
    },
    /// An update started, in its generation or its epoch form: the report is
    /// the update as it stood when its `Preparing` began.
    Update(Box<UpdateReport>),
}

/// A module set as a report names it: the engine package, the kernel package of each slot and
/// the scheduling policy package.
pub(crate) fn modules_report(modules: &crate::epoch::ModuleSet) -> Modules {
    Modules {
        engine: modules.engine.clone(),
        kernels: crate::epoch::kernel_packages(&modules.kernels),
        policy: modules.policy.clone(),
    }
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
