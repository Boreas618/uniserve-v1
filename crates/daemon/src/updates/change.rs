//! What applying a configuration change did (`Leader::apply`), which the node records in the
//! change's revision.

use proto::admin::{InPlaceChange, UpdateReport};

use crate::config::Config;

/// What an apply did.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Applied {
    /// The target is the current configuration.
    Unchanged,
    /// The replicas took the options that changed at their engines' next step, or the
    /// daemon reads them from its next request on.
    InPlace(Vec<InPlaceChange>),
    /// The daemon's own options change: a hot restart of the daemon serving `config`, the
    /// current configuration with the target's daemon options, applies them, and the successor
    /// applies the change again for the rest: the daemon changes before what it updates.
    Daemon { config: Box<Config> },
    /// An update started, in its generation or its epoch form: the report is
    /// the update as it stood when its `Preparing` began.
    Update(Box<UpdateReport>),
}
