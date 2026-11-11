//! The kernel installs of a precopy: the Build Server's selection of the generation's packages
//! for every key the devices launch, then one install per device admitted on the module-load lane
//! with every selected slot and exactly its keys.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use build_server::{Descriptors, PackageRequest, PrepareRequest, Selector, Target};
use ipc::UpdateFailure;
use proto::module::Packages;
use worker::host::HostTask;

use super::MainProcess;
use crate::modules::load::{Install, ModuleWork, Select};
use crate::modules::select::Routes;

impl MainProcess {
    /// Every descriptor key this node's generation launches, by operator: the union of the local
    /// workers' recorded walks, and of what the serving generation holds.
    ///
    /// The recorded walk (`worker::runners::recording`) states every key a device's forwards
    /// launch before anything is installed, so the epoch's first generation is installed from it in
    /// one admission. An update changes implementations and never the set of descriptors, and the
    /// serving generation's own keys are the same set as the walk's.
    pub(super) fn generation_keys(&self) -> BTreeMap<String, BTreeSet<String>> {
        let mut keys: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for worker in &self.workers {
            for (operator, recorded) in worker.recorded() {
                keys.entry(operator.clone())
                    .or_default()
                    .extend(recorded.iter().cloned());
            }
        }
        if let Some(serving) = self.update.generations.first() {
            for (operator, key, _) in serving.iter() {
                keys.entry(operator.to_owned())
                    .or_default()
                    .insert(key.to_owned());
            }
        }
        keys
    }

    /// Admits the Build Server's selection of the generation's packages on the module-load lane:
    /// every operator's keys, with the backends the deployment names for it.
    pub(super) fn admit_selection(
        &mut self,
        named: &Packages,
        keys: BTreeMap<String, BTreeSet<String>>,
    ) -> std::result::Result<Arc<HostTask<ModuleWork>>, UpdateFailure> {
        for operator in keys.keys() {
            if proto::module::operator::named(operator).is_none() {
                return Err(UpdateFailure::Precopy {
                    reason: format!("operator {operator} has no implementation in this build"),
                });
            }
        }
        let select = Select {
            packages: Arc::clone(&self.packages),
            targets: self.targets.clone(),
            named: named.clone(),
            keys,
        };
        self.modules
            .admit(ModuleWork::Select(select))
            .map_err(|error| UpdateFailure::Precopy {
                reason: format!("admitting the selection of the generation's packages: {error}"),
            })
    }

    /// The module-load tasks one generation needs, one per device: each is one `prepare` of every
    /// slot in `packages`, its package asked for exactly the keys `routes` selected for it.
    pub(super) fn admit_installs(
        &mut self,
        packages: &Packages,
        routes: &Routes,
    ) -> std::result::Result<Vec<Arc<HostTask<ModuleWork>>>, UpdateFailure> {
        let invalid = |reason: String| UpdateFailure::Precopy { reason };
        let mut entries = Vec::with_capacity(packages.len());
        let mut descriptors = Vec::with_capacity(packages.len());
        for (slot, package) in packages {
            entries.push(PackageRequest {
                slot: slot.clone(),
                selector: Selector::Fixed(package.clone()),
            });
            descriptors.push(Descriptors {
                slot: slot.clone(),
                keys: routes.get(slot).cloned().unwrap_or_default(),
            });
        }

        let mut tasks = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            let ordinal = worker.device();
            let device = platform::Device::new(ordinal)
                .map_err(|error| invalid(format!("opening device {}: {error}", ordinal.0)))?;
            // `targets` holds each distinct architecture once, so a device's own is read from it.
            let target = ordinal
                .compute_capability()
                .map(Target::arch)
                .map_err(|error| invalid(format!("reading device {}: {error}", ordinal.0)))?;
            if !self.targets.contains(&target) {
                return Err(invalid(format!("device {} has no build target", ordinal.0)));
            }
            let install = Install {
                packages: Arc::clone(&self.packages),
                device,
                request: PrepareRequest {
                    targets: vec![target.clone()],
                    packages: entries.clone(),
                    descriptors: descriptors.clone(),
                },
                target,
            };
            match self.modules.admit(ModuleWork::Install(install)) {
                Ok(task) => tasks.push(task),
                Err(error) => {
                    return Err(invalid(format!("admitting a module load: {error}")));
                }
            }
        }
        Ok(tasks)
    }
}
