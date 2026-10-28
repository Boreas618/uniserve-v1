//! An in-place weight update (`uniserve.update_weights`): the rank rewrites
//! its loaded tensors where they lie from another checkpoint of the same
//! model, or from a plugin's weight source that streams its tensors, and
//! renames its retained weights after it.
//!
//! The library loads the checkpoint into new storage as the rank loaded its
//! model and copies every tensor into place once all of them match
//! (`uniserve.loading.rank.update_weights`); the engine runs the update
//! while the rank sleeps, so the KV cache's storage holds the new copy. The
//! retained weights are mapped read-only, so the rank maps them writable
//! for the copy and read-only again after. Their keys digest the checkpoint
//! they were loaded from (`weights`): the Storage Server moves them to the
//! keys of the new checkpoint, which a relaunch on it binds. The rank's
//! cache pages are keyed by the model's fingerprint too, so the update
//! answers the new key seed of each host page pool, which the engine takes.

use std::collections::BTreeMap;
use std::sync::Arc;

use cuda::Protection;
use tvm_ffi::Any;

use super::loading::{Description, LoadedModel};
use crate::config::{ExpertUnion, WorkerLaunch};
use crate::ffi::{Backing, Handle, Interpreter, MemoryKind};
use crate::parallel::{DwdpUpdate, ExpertWeights};
use crate::storage::CachePlan;
use crate::storage::allocation::Storage;
use crate::storage::prefix_cache::host::page_contents;
use crate::storage::retained::Retained;
use crate::storage::weights::{self, Planned, Rebalance, WeightKeys};
use crate::{Error, Result};
use ::storage::manifest::weights::{Sources, Transfer};

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
    /// Runtime-owned expert routing, bound before numerical graph capture.
    pub(crate) routing: Handle,
    /// Creating handles of the allocations currently mapped at graph addresses.
    pub(crate) allocations: BTreeMap<::storage::Key, Arc<cuda::PhysicalAllocation>>,
    /// Background weight redistribution, before the common commit boundary.
    pub(crate) rebalance: Option<ExpertUpdate>,
    pub(crate) distributed: Option<Arc<DwdpUpdate>>,
    pub(crate) host_paths: Vec<String>,
}

pub(crate) struct ExpertUpdate {
    copies: ExpertCopies,
    orders: BTreeMap<String, Vec<u32>>,
    selected: String,
}

enum ExpertCopies {
    Planned {
        keys: WeightKeys,
        layout: Planned,
        sources: Sources,
        transfer: Box<Transfer>,
    },
    Copying(Rebalance),
    Preparing(std::thread::JoinHandle<Result<(Rebalance, ExpertWeights)>>),
    Ready(Rebalance, ExpertWeights),
}

impl WeightUpdate {
    fn expert_layout(&self, orders: &str) -> Result<Handle> {
        let resident = self
            .distributed
            .as_ref()
            .map(|owner| owner.layout(orders))
            .transpose()?;
        Handle::new(
            Interpreter::import("uniserve.bindings.routing.ExpertRouting.describe")?.call_values(
                &[
                    self.routing.to_any(),
                    self.model.to_any(),
                    text(orders),
                    Any::from(crate::ffi::strings(&self.host_paths)),
                    resident.as_ref().map(Handle::to_any).unwrap_or_default(),
                ],
            )?,
        )
    }
    /// Plan receiver-driven copies without allocating their destination storage.
    pub(crate) fn prepare_experts(
        &mut self,
        orders: BTreeMap<String, Vec<u32>>,
        retained: &Retained,
    ) -> Result<serde_json::Value> {
        if self.rebalance.is_some() {
            return Err(Error::State("an expert redistribution is already prepared"));
        }
        let mut launch = self.launch.clone();
        launch.expert_order.extend(orders);
        let encoded = serde_json::to_string(&launch.expert_order)
            .map_err(|error| Error::Invalid(error.to_string()))?;
        let selected: tvm_ffi::String =
            Interpreter::import("uniserve.bindings.routing.ExpertRouting.select")?
                .call_values(&[self.routing.to_any(), text(&encoded)])?
                .try_into()
                .map_err(|error| Error::numerical("expert order", &error))?;
        let layout = self.expert_layout(selected.as_str())?;
        let previous = self.keys.as_ref().ok_or(Error::State(
            "expert weights must be retained before redistribution",
        ))?;
        let current = serde_json::to_string(&self.launch.expert_order)
            .map_err(|error| Error::Invalid(error.to_string()))?;
        let current_layout = self.expert_layout(&current)?;
        self.allocations.extend(
            previous
                .plan_layout(current_layout)?
                .allocations(&self.storages)?,
        );
        let description = Description::describe(&launch)?;
        let fingerprints = LoadedModel::fingerprint(&description, &launch)?
            .ok_or(Error::State("expert weights name their checkpoint sources"))?;
        let devices: Vec<_> = launch
            .execution
            .devices()
            .filter_map(|device| device.cuda())
            .collect();
        let keys = WeightKeys::new(
            retained,
            &fingerprints,
            &weights::position(&launch)?,
            &devices,
            &super::chunk_degrees(&launch)?,
        )?;
        let planned = keys.plan_layout(layout)?;
        let mut sources = keys
            .sources(retained, &launch.storage.peers, Some(&[retained.world()]))?
            .ok_or(Error::State(
                "the serving world's expert weights are retained",
            ))?;
        let transfer = keys.plan_rebalance(retained, &planned, &mut sources)?;
        let bytes = transfer.manifest.created_bytes(&transfer.request, 0)
            + self
                .distributed
                .as_ref()
                .map_or(0, |owner| owner.required_bytes() as u64);
        let gpu = transfer.request.gpus[0];
        let local: serde_json::Value = serde_json::from_str(selected.as_str())
            .map_err(|error| Error::Interface(error.to_string()))?;
        self.rebalance = Some(ExpertUpdate {
            copies: ExpertCopies::Planned {
                keys,
                layout: planned,
                sources,
                transfer: Box::new(transfer),
            },
            orders: launch.expert_order,
            selected: selected.as_str().to_owned(),
        });
        Ok(serde_json::json!({"orders": local, "gpu": gpu, "bytes": bytes}))
    }

    /// Allocate and copy only after the coordinator obtained the planned room.
    pub(crate) fn start_experts(&mut self, retained: &Arc<Retained>) -> Result<()> {
        let mut update = self
            .rebalance
            .take()
            .ok_or(Error::State("no expert redistribution is prepared"))?;
        let ExpertCopies::Planned {
            keys,
            layout,
            sources,
            transfer,
        } = update.copies
        else {
            self.rebalance = Some(update);
            return Err(Error::State("expert copies have already started"));
        };
        let previous = self
            .keys
            .as_ref()
            .ok_or(Error::State("the serving weights are retained"))?;
        let transfer = keys.rebalance(
            retained,
            layout,
            &self.storages,
            previous,
            &sources,
            *transfer,
        )?;
        update.copies = match &self.distributed {
            Some(owner) => {
                let owner = Arc::clone(owner);
                let retained = Arc::clone(retained);
                ExpertCopies::Preparing(std::thread::spawn(move || {
                    let result = (|| {
                        let mut transfer = transfer;
                        let bound = transfer.wait(&retained)?.clone();
                        let successor = owner.prepare(&bound)?;
                        Ok((transfer, successor))
                    })();
                    result.map_err(|error: Error| {
                        owner.cancel();
                        Error::Invariant(format!("distributed expert preparation failed: {error}"))
                    })
                }))
            }
            None => ExpertCopies::Copying(transfer),
        };
        self.rebalance = Some(update);
        Ok(())
    }

    pub(crate) fn experts_ready(&mut self, retained: &Retained) -> Result<bool> {
        let mut update = self
            .rebalance
            .take()
            .ok_or(Error::State("no expert redistribution is prepared"))?;
        let (copies, ready) = match update.copies {
            ExpertCopies::Preparing(task) if task.is_finished() => {
                let (transfer, successor) = task.join().map_err(|_| {
                    Error::Invariant("distributed expert preparation panicked".into())
                })??;
                (ExpertCopies::Ready(transfer, successor), true)
            }
            ExpertCopies::Copying(mut transfer) => {
                let ready = transfer.ready(retained);
                update.copies = ExpertCopies::Copying(transfer);
                self.rebalance = Some(update);
                return ready;
            }
            copies @ ExpertCopies::Ready(..) => (copies, true),
            copies => (copies, false),
        };
        update.copies = copies;
        self.rebalance = Some(update);
        Ok(ready)
    }

    /// Publish weight mappings and tensor routing at the drained boundary.
    pub(crate) fn commit_experts(&mut self, retained: &Retained) -> Result<()> {
        let update = self
            .rebalance
            .take()
            .ok_or(Error::State("no expert redistribution is prepared"))?;
        let (transfer, resident) = match update.copies {
            ExpertCopies::Copying(transfer) => (transfer, None),
            ExpertCopies::Ready(transfer, successor) => {
                let owner = self
                    .distributed
                    .as_ref()
                    .ok_or(Error::State("distributed expert weights have an owner"))?;
                owner.commit(successor)?;
                (
                    transfer,
                    Some((owner.layout(&update.selected)?, owner.storages()?)),
                )
            }
            _ => return Err(Error::State("expert copies have not completed")),
        };
        let (keys, storages) = transfer.commit(retained, &mut self.allocations, resident)?;
        Interpreter::import("uniserve.bindings.routing.ExpertRouting.commit")?
            .call_values(&[self.routing.to_any(), text(&update.selected)])
            .map_err(|error| {
                Error::Invariant(format!("expert routing could not be published: {error}"))
            })?;
        self.keys = Some(keys);
        self.storages = storages;
        self.launch.expert_order = update.orders;
        Ok(())
    }

    pub(crate) fn abort_experts(&mut self, retained: &Retained) -> Result<()> {
        if let Some(update) = self.rebalance.take() {
            match update.copies {
                ExpertCopies::Copying(transfer) | ExpertCopies::Ready(transfer, _) => {
                    transfer.abort(retained)?
                }
                ExpertCopies::Preparing(_) => {
                    if let Some(owner) = &self.distributed {
                        owner.cancel();
                    }
                    return Err(Error::Invariant(
                        "interrupted distributed expert preparation requires group recovery".into(),
                    ));
                }
                ExpertCopies::Planned { .. } => {}
            }
        }
        Ok(())
    }

    /// Why the rank's weights cannot be rewritten in place, if so: stacks
    /// streamed from host storage lie in sealed host storage, and DWDP moves
    /// the routed experts into storage of the expert union.
    fn refusal(&self) -> Option<&'static str> {
        if self
            .launch
            .components
            .values()
            .any(|component| component.weights == ipc::ComponentWeights::Host)
        {
            return Some("weights streamed from host storage lie in sealed storage");
        }
        if self
            .launch
            .expert_parallel
            .as_ref()
            .is_some_and(ExpertUnion::prefetches_weights)
        {
            return Some("DWDP keeps the routed experts in storage of the expert union");
        }
        None
    }

    /// The retained storages in device storage, which the rank maps
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
            let renamed = keys.under(fingerprints);
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

fn text(value: &str) -> Any {
    Any::from(tvm_ffi::String::from(value))
}
