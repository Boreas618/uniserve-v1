//! Where each live rank incarnation sits in the deployment.

use std::collections::HashMap;
use std::sync::{Arc, PoisonError, RwLock};

use uniserve_ipc::WorkerEndpoint;

use crate::WorkerGroupId;

/// The worker and rank each live rank incarnation was launched as.
///
/// A rank reports only its own incarnation: the engine chose which worker and
/// rank that process serves when it launched it, so the engine is the only
/// owner of that fact. Groups register their ranks when the deployment is
/// bound and again when a group installs replacement ranks, which retires the
/// incarnations they replace; a group that loses its ranks retires them at
/// once. Readers resolve the incarnation a publication names, and a
/// publication of a retired incarnation resolves to nothing.
///
/// Clones share one table.
#[derive(Debug, Clone, Default)]
pub(crate) struct IncarnationRegistry {
    positions: Arc<RwLock<HashMap<String, (WorkerGroupId, u32)>>>,
}

impl IncarnationRegistry {
    /// Makes `endpoints`, in rank order, the live incarnations of `worker`.
    ///
    /// Every incarnation `worker` held before is retired.
    pub(crate) fn assign<'a>(
        &self,
        worker: &WorkerGroupId,
        endpoints: impl IntoIterator<Item = &'a WorkerEndpoint>,
    ) {
        let mut positions = self
            .positions
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        positions.retain(|_, (owner, _)| owner != worker);
        for (rank, endpoint) in endpoints.into_iter().enumerate() {
            positions.insert(endpoint.incarnation.clone(), (worker.clone(), rank as u32));
        }
    }

    /// Returns the worker and rank `endpoint`'s incarnation serves, or `None`
    /// when it is not a live incarnation of this deployment.
    pub(crate) fn position(&self, endpoint: &WorkerEndpoint) -> Option<(WorkerGroupId, u32)> {
        self.positions
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&endpoint.incarnation)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(incarnation: &str) -> WorkerEndpoint {
        WorkerEndpoint {
            host: "host".to_owned(),
            address_space: format!("{incarnation}-process"),
            incarnation: incarnation.to_owned(),
        }
    }

    /// Assigning a worker's replacement ranks retires exactly that worker's
    /// earlier incarnations, and another worker's stay live.
    #[test]
    fn replacement_retires_the_replaced_incarnations_of_that_worker_only() {
        let incarnations = IncarnationRegistry::default();
        let text = WorkerGroupId("text".to_owned());
        let decoder = WorkerGroupId("decoder".to_owned());
        let before = [endpoint("text-0"), endpoint("text-1")];
        let other = endpoint("decoder-0");
        incarnations.assign(&text, &before);
        incarnations.assign(&decoder, [&other]);
        assert_eq!(incarnations.position(&before[1]), Some((text.clone(), 1)));

        let after = [endpoint("text-0b"), endpoint("text-1b")];
        incarnations.assign(&text, &after);

        assert_eq!(incarnations.position(&before[0]), None);
        assert_eq!(incarnations.position(&before[1]), None);
        assert_eq!(incarnations.position(&after[1]), Some((text, 1)));
        assert_eq!(incarnations.position(&other), Some((decoder, 0)));
    }
}
