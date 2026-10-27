//! An in-place weight update (`uniserve.update_weights`): the rank rewrites
//! its loaded tensors where they lie from another checkpoint of the same
//! model, or from a plugin's weight source that streams its tensors, and
//! renames its retained weights after it.
//!
//! The library loads the checkpoint into new memory as the rank loaded its
//! model and copies every tensor into place once all of them match
//! (`uniserve.loading.rank.update_weights`); the engine runs the update
//! while the rank sleeps, so the KV cache's memory holds the new copy. The
//! retained weights are mapped read-only, so the rank maps them writable
//! for the copy and read-only again after. Their keys digest the checkpoint
//! they were loaded from (`weights`): the Storage Server moves them to the
//! keys of the new checkpoint, which a relaunch on it binds. The rank's
//! cache pages are keyed by the model's fingerprint too, so the update
//! answers the new key seed of each host page pool, which the engine takes.

use std::sync::Arc;

use cuda::{DeviceId, Protection};
use tvm_ffi::Any;

use super::loading::{Description, LoadedModel};
use crate::config::{ExpertUnion, WorkerLaunch};
use crate::ffi::{Backing, Handle, Interpreter, MemoryKind};
use crate::memory::CachePlan;
use crate::memory::allocation::Storage;
use crate::memory::prefix_cache::host::page_contents;
use crate::memory::retained::Retained;
use crate::memory::weights::{self, WeightKeys};
use crate::{Error, Result};

/// What an in-place weight update rewrites and renames.
pub(crate) struct WeightUpdate {
    /// The launch the rank loaded its model with, naming the checkpoint
    /// its weights are from.
    pub(crate) launch: WorkerLaunch,
    /// The loaded model (`uniserve.loading.rank.Loaded`).
    pub(crate) model: Handle,
    /// The retained storages the weights lie in.
    pub(crate) storages: Vec<Arc<Storage>>,
    /// Their keys; `None` for weights the rank does not retain.
    pub(crate) keys: Option<WeightKeys>,
    /// The identity of what the model's forwards compute
    /// (`weights::model_identity`), which keys its cache pages with the
    /// rewritten weights' fingerprint.
    pub(crate) identity: String,
}

impl WeightUpdate {
    /// Why the rank's weights cannot be rewritten in place, if so: stacks
    /// streamed from host memory lie in sealed host memory, and DWDP moves
    /// the routed experts into memory of the expert union.
    fn refusal(&self) -> Option<&'static str> {
        if self
            .launch
            .components
            .values()
            .any(|component| component.weights == ipc::ComponentWeights::Host)
        {
            return Some("weights streamed from host memory lie in sealed memory");
        }
        if self
            .launch
            .expert_parallel
            .as_ref()
            .is_some_and(ExpertUnion::prefetches_weights)
        {
            return Some("DWDP keeps the routed experts in memory of the expert union");
        }
        None
    }

    /// The retained storages in device memory, which the rank maps
    /// read-only once published.
    fn device_storages(&self) -> impl Iterator<Item = &Arc<Storage>> {
        self.storages
            .iter()
            .filter(|storage| matches!(Backing::device(storage.as_ref()), MemoryKind::Cuda(_)))
    }

    /// Rewrite the weights from the checkpoint at `path`, and answer this
    /// rank's worker group and the new key seed of each of its host page
    /// pools (`plan`'s groups, when the rank copies pages through the host
    /// KV cache). A rank without checkpoint weights answers no seeds.
    pub(crate) fn apply(
        &mut self,
        path: &str,
        retained: &Retained,
        plan: Option<&CachePlan>,
        host_cache: bool,
    ) -> Result<serde_json::Value> {
        if let Some(reason) = self.refusal() {
            return Err(Error::Unsupported(format!(
                "the weights cannot be updated in place: {reason}; deploy the new checkpoint instead"
            )));
        }
        for storage in self.device_storages() {
            storage.protect(Protection::ReadWrite)?;
        }
        let updated =
            Interpreter::import("uniserve.loading.rank.update_weights")?.call_as::<bool>(&[
                self.model.view(),
                (&Any::from(tvm_ffi::String::from(path))).into(),
            ]);
        for storage in self.device_storages() {
            storage.protect(Protection::ReadOnly)?;
        }
        let group = self.launch.group.clone();
        if !updated? {
            return Ok(serde_json::json!({ "group": group, "page_contents": [] }));
        }

        let mut launch = self.launch.clone();
        let model = launch.model.as_mut().ok_or(Error::State(
            "a rank with checkpoint weights names its model",
        ))?;
        // A plugin's weight source streams the tensors of the same
        // checkpoint; a directory is another checkpoint.
        if path.contains("://") {
            model.weights = Some(path.to_owned());
        } else {
            model.path = path.to_owned();
            model.weights = None;
        }
        let description = Description::describe(&launch)?;
        let fingerprints = LoadedModel::fingerprint(&description, &launch)?;
        if let (Some(keys), Some(fingerprints)) = (&self.keys, &fingerprints) {
            let devices: Vec<DeviceId> = launch
                .execution
                .devices()
                .filter_map(|device| device.cuda())
                .collect();
            let renamed = WeightKeys::new(
                retained,
                fingerprints,
                &weights::placement(&launch)?,
                &devices,
            )?;
            let moves = keys.moves(&renamed, retained)?;
            let moved = moves.len();
            retained.rekey(moves)?;
            tracing::info!(moved, "renamed the retained weights after the checkpoint");
            self.keys = Some(renamed);
        }
        self.launch = launch;
        let contents: Vec<String> = match (plan, &fingerprints) {
            (Some(plan), Some(fingerprints)) if host_cache => {
                page_contents(plan, &fingerprints.pages(&self.identity))
                    .iter()
                    .map(|content| content.iter().map(|byte| format!("{byte:02x}")).collect())
                    .collect()
            }
            _ => Vec::new(),
        };
        // The identity that roots the device pages' hash chains from now on.
        let identity = fingerprints
            .as_ref()
            .map(|fingerprints| fingerprints.pages(&self.identity));
        Ok(serde_json::json!({ "group": group, "page_contents": contents, "identity": identity }))
    }
}
