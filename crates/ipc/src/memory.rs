//! What a rank reports of the memory operations the engine runs on it.

use serde::{Deserialize, Serialize};

/// What a rank responds of a weight update.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightsUpdated {
    /// The rank's worker group and rank; `None` from a worker of no
    /// storage scope.
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub rank: Option<u32>,
    /// The identity of the rank's new weights, which keys the trajectories
    /// they compute.
    #[serde(default)]
    pub identity: Option<String>,
    /// The page identity of each of the rank's cache groups under the new
    /// weights (`PrefixCacheInfo::identities`); empty for a rank that keeps
    /// no paged cache.
    #[serde(default)]
    pub identities: Vec<String>,
    /// The weights the update rewrote.
    #[serde(default)]
    pub moved: usize,
}
