//! The deployment's updates: an epoch switch and the trial that decides it
//! (`specs/worker.md` §7.4, §7.3, §2.5).
//!
//! An update that changes what builds a main process re-builds the epoch
//! ([`crate::epoch::layer`] returns [`crate::epoch::Layer::Epoch`]): each
//! node's main process of the target executable starts as a parked standby,
//! the serving epoch is drained and the target serves, the trial decides, and
//! a commit or a revert follows. Participants are node main processes, not an
//! engine and its workers, and the wire between them and the node's daemon is
//! `proto::process` ([`crate::proto::process`]): commands down, reports up,
//! the roster of children behind them.
//!
//! [`switch`] holds the switch itself: the phases `Preparing`, `Promoting`,
//! `Serving`, `Probation`, `Committing` or `Reverting`, and `Purged` or
//! `Abandoned`; what ends each one (each node's `Parked`, `Promoted`,
//! `Drained`, `Committed` or `Reverted`); the pause from the first `Promote`
//! to the last drained epoch; and the commit, revert and purge it commands.
//!
//! [`change`] holds the shapes of a configuration change ([`change::
//! ApplyRequest`], [`change::Applied`], [`change::InFlight`]), which the API
//! and the leader use and the epoch layer reads before it starts a switch.
//!
//! The update entry (`impl Leader`'s `apply`, `commit`, `rollback`,
//! `config_status`) is still `worlds.rs`'s while the landing order moves the
//! type definitions first; the switch here is driven by it once that entry
//! moves.

use crate::leader::UpdateError;
use crate::options::ServeConfig;

pub(crate) mod change;
pub(crate) mod epoch;
pub(crate) mod switch;

use change::ApplyRequest;

/// Resolves an apply request against the serving configuration: the target
/// configuration, the fields the request stated, and the target's update
/// options. Compares nothing and applies nothing — [`crate::epoch::layer`]
/// and the update entry decide what the target means.
///
/// The API and the leader share this one resolution, so the target an
/// operator sees planned is the target the leader applies.
pub(crate) fn resolve(
    current: &ServeConfig,
    request: &ApplyRequest,
) -> Result<
    (
        ServeConfig,
        crate::options::Stated,
        crate::options::UpdateOptions,
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
