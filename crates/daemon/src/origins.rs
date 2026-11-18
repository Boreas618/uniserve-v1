//! Where the modules each replica runs come from: the `uniserve_replica_module` series of the
//! observation store (`specs/configuration.md` §9, item 13).
//!
//! The Build Server records with every installed revision the git commit of the tree it was
//! installed from and whether that tree was clean (`build_server::inputs::Origin`). The leader
//! states, for every replica and every module of its module set (the engine, the scheduling
//! policy and each kernel package), a gauge of 1 labelled with the module's interface, package,
//! revision and that origin, so a query finds the replicas that run a revision no release holds:
//! one of no commit, or of a tree that was not clean. A module a replica no longer runs reads 0
//! from the next record on.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Mutex;

use anyhow::Context as _;
use proto::module::{ModuleSet, PackageRevision};

use crate::observation::{Labels, Store};

/// The series' name.
pub(crate) const METRIC: &str = "uniserve_replica_module";

/// The label sets the last record wrote at 1, which the next one sets to 0 when they no longer
/// hold.
#[derive(Debug, Default)]
pub(crate) struct Origins {
    written: Mutex<BTreeSet<Labels>>,
}

impl Origins {
    /// Records the modules each replica of `replicas` (its ordinal and its module set) runs at
    /// configuration revision `revision`, with their origins as the Build Server at `socket`
    /// lists them. Blocks on the Build Server.
    ///
    /// # Errors
    ///
    /// Fails when the Build Server cannot be reached or does not list, or the store refuses a
    /// value.
    pub(crate) fn record(
        &self,
        store: &Store,
        revision: u64,
        socket: &Path,
        replicas: &[(usize, ModuleSet)],
    ) -> anyhow::Result<()> {
        let mut client = build_server::Client::connect(socket)
            .with_context(|| format!("connecting to the Build Server at {}", socket.display()))?;
        let origins: BTreeMap<(String, String, u32), build_server::inputs::Origin> = client
            .list()
            .context("listing the Build Server's packages")?
            .into_iter()
            .map(|(package, origin, _)| ((package.op, package.backend, package.revision), origin))
            .collect();

        let mut rows = BTreeSet::new();
        for (replica, modules) in replicas {
            for (op, package) in packages(modules) {
                let origin = origins
                    .get(&(op.to_owned(), package.backend.clone(), package.revision))
                    .cloned()
                    .unwrap_or_default();
                rows.insert(Labels::from([
                    ("replica".to_owned(), replica.to_string()),
                    ("op".to_owned(), op.to_owned()),
                    ("backend".to_owned(), package.backend.clone()),
                    ("revision".to_owned(), package.revision.to_string()),
                    ("commit".to_owned(), origin.commit.unwrap_or_default()),
                    ("clean".to_owned(), origin.clean.to_string()),
                ]));
            }
        }

        let mut written = service::lock(&self.written);
        for retired in written.difference(&rows) {
            store.set(METRIC, retired, revision, 0.0)?;
        }
        for row in &rows {
            store.set(METRIC, row, revision, 1.0)?;
        }
        *written = rows;
        Ok(())
    }
}

/// Every package of `modules` a replica runs, with the interface it fills.
fn packages(modules: &ModuleSet) -> impl Iterator<Item = (&str, &PackageRevision)> {
    [
        (proto::module::ENGINE, &modules.engine),
        (proto::module::PROGRAM, &modules.policy),
    ]
    .into_iter()
    .chain(
        modules
            .kernels
            .iter()
            .map(|(slot, package)| (slot.op.as_str(), package)),
    )
}
