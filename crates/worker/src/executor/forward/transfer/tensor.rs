//! Tensor product transfers (`TransferMode::Tensor`).
//!
//! Each active call copies its one input product into its one output
//! product in the batch's job and exports the copy, which is published at
//! the job's completion. The input may be a
//! product of this rank or one the batch imported from another rank or
//! worker; either way it is leased from the tensor store. The output keeps
//! its kind: a relayed scalar (a sampled token) lands in its request's
//! relay lane, an encoder feature in a feature allocation with its spatial
//! metadata, any other product in its scheduler allocation. A transfer
//! never converts: the output declares the input's dtype.
//!
//! A trajectory transfer (a latent input and output) instead exports the
//! committed version of the call's request trajectory from the latent pool
//! as its output, which other ranks and workers import into their own
//! pools ([`super::latents`]).

use std::collections::HashMap;

use indexmap::IndexMap;
use uniserve_ipc::{BufferAllocation, BufferId, DType as ProductDType, TensorRef};

use super::batch::TransferBatch;
use super::latents;
use crate::executor::forward::batch::image::ImageParts;
use crate::ffi::Device;
use crate::storage::memory::parse_dtype;
use crate::storage::{OutputPlacement, ProductBuffer};
use crate::worker::RankStorage;
use crate::{Error, Result};

/// How a transfer's output product is stored.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Placement {
    /// An encoder feature in a feature allocation.
    Feature,
    /// A product in its scheduler allocation.
    Persistent,
    /// A scalar in its request's relay lane.
    Relay,
}

impl TransferBatch<'_, '_> {
    /// Copy every active call's input product into its output product and
    /// export the copy; export trajectory transfers' versions from the
    /// latent pool of `image`.
    pub(super) fn copy_products(&mut self, image: Option<&mut ImageParts<'_>>) -> Result<()> {
        let plan = self.plan;
        let target = self.target();
        let allocations: HashMap<BufferId, BufferAllocation> = plan
            .buffer_allocations
            .iter()
            .map(|allocation| (allocation.buffer, *allocation))
            .collect();

        // Each active call moves one product into one output of the same
        // physical kind.
        let mut transfers = Vec::new();
        let mut trajectories = Vec::new();
        for index in self.active() {
            let call = &plan.calls[index];
            if call.latent_input.is_some() || call.latent_output.is_some() {
                trajectories.push(index);
                continue;
            }
            let mut inputs = call.tensor_inputs();
            let mut outputs = call.tensor_outputs();
            let (Some(input), None, Some(output), None) =
                (inputs.next(), inputs.next(), outputs.next(), outputs.next())
            else {
                return Err(Error::Invalid(
                    "product transfer requires one physical input and one output".into(),
                ));
            };
            let placement = if call.encoder_output.as_ref() == Some(output) {
                Placement::Feature
            } else if allocations.contains_key(&output.buffer_id()) {
                Placement::Persistent
            } else {
                Placement::Relay
            };
            transfers.push((index, input, output, placement));
        }
        if !trajectories.is_empty() {
            let image = image.ok_or_else(|| {
                Error::Unsupported("trajectory transfers require a latent pool".into())
            })?;
            self.export_trajectories(image, &trajectories)?;
        }
        let writes = self.bind(&transfers, &allocations, target)?;

        let issue = self.issue();
        for ((index, input, output, _), write) in transfers.into_iter().zip(writes) {
            let call = &plan.calls[index];
            let read = self
                .storage
                .tensors
                .consume(input, call.call_id, Some(target), issue)?;
            self.reads.push(std::sync::Arc::clone(&read));
            if read.value().dtype() != product_dtype(output.dtype)? {
                return Err(Error::Invalid(
                    "product transfer changes its declared representation".into(),
                ));
            }
            self.storage
                .tensors
                .write(&write, read.value(), issue, read.metadata())?;
            let RankStorage {
                tensors,
                transports,
                ..
            } = &*self.storage;
            let export =
                tensors.export_buffer(call, output, &write, transports, false, None, issue)?;
            self.recorded.products.push_back((index, write, export));
        }
        Ok(())
    }

    /// Export the committed trajectory version each of `calls` names (its
    /// latent input) as its latent output, ordered after the latent pool's
    /// previous user in the batch's job.
    fn export_trajectories(&mut self, image: &mut ImageParts<'_>, calls: &[usize]) -> Result<()> {
        let plan = self.plan;
        let issue = self.issue();
        image.enter_latents(issue)?;
        let exported = (|| -> Result<()> {
            for &index in calls {
                let call = &plan.calls[index];
                let (Some(input), Some(output)) = (&call.latent_input, &call.latent_output) else {
                    return Err(Error::Invalid(
                        "product transfer changes the physical product kind".into(),
                    ));
                };
                if call.tensor_inputs().count() != 1 || call.tensor_outputs().count() != 1 {
                    return Err(Error::Invalid(
                        "product transfer requires one physical input and one output".into(),
                    ));
                }
                let params = plan
                    .latent_params
                    .iter()
                    .find(|params| {
                        params.request_key == call.request_key && params.call_id == call.call_id
                    })
                    .ok_or_else(|| {
                        Error::Invalid("latent transfer has no bound parameters".into())
                    })?;
                let slot = self.recorded.calls[index].slot;
                let (pool, storage) = image.latent_pool()?;
                let pages: Vec<usize> = params
                    .page_table
                    .iter()
                    .map(|&page| page as usize)
                    .collect();
                let bank = pool.current_bank(
                    slot,
                    i64::from(params.start_step),
                    i64::from(input.generation),
                    i64::from(params.latent_units),
                    i64::from(params.height),
                    i64::from(params.width),
                    &pages,
                )?;
                let version = latents::issue_export(
                    pool,
                    storage,
                    &self.storage.transports,
                    call,
                    output,
                    slot,
                    params,
                    bank,
                    params.start_step,
                    issue,
                )?;
                self.recorded.versions.push_back((index, version));
            }
            Ok(())
        })();
        exported.and(image.leave_latents(issue))
    }

    /// Bind every transfer's output, in transfer order: features in their
    /// allocations, then persistent products and relayed scalars as one
    /// atomic group binding (scalars grouped by dtype and shape).
    fn bind(
        &mut self,
        transfers: &[(usize, &TensorRef, &TensorRef, Placement)],
        allocations: &HashMap<BufferId, BufferAllocation>,
        target: Device,
    ) -> Result<Vec<ProductBuffer>> {
        let features: Vec<(TensorRef, Device)> = transfers
            .iter()
            .filter(|transfer| transfer.3 == Placement::Feature)
            .map(|transfer| (transfer.2.clone(), target))
            .collect();
        let mut bound = self.storage.tensors.reserve_features(
            &features,
            allocations,
            OutputPlacement::default(),
        )?;
        self.recorded.writes.extend(bound.iter().cloned());

        let mut groups: IndexMap<(Placement, ProductDType, String), Vec<(TensorRef, Device)>> =
            IndexMap::new();
        for &(_, _, output, placement) in transfers {
            if placement == Placement::Feature {
                continue;
            }
            groups
                .entry((placement, output.dtype, format!("{:?}", output.shape_bound)))
                .or_default()
                .push((output.clone(), target));
        }
        let mut persistent = Vec::new();
        let mut relays = Vec::new();
        for ((placement, _, _), group) in groups {
            match placement {
                Placement::Persistent => persistent.extend(group),
                _ => relays.push(group),
            }
        }
        let mut grouped = relays;
        if !persistent.is_empty() {
            grouped.push(persistent);
        }
        let slots = self.slots();
        let others = self.storage.tensors.bind_output_groups(
            &grouped,
            OutputPlacement {
                request_slots: Some(&slots),
                allocations: Some(allocations),
                ..OutputPlacement::default()
            },
        )?;
        for write in others.into_iter().flatten() {
            self.recorded.writes.push(write.clone());
            bound.push(write);
        }

        // Pair each transfer with its write by buffer identity.
        let mut writes: HashMap<BufferId, ProductBuffer> =
            bound.into_iter().map(|write| (write.id(), write)).collect();
        transfers
            .iter()
            .map(|transfer| {
                writes
                    .remove(&transfer.2.buffer_id())
                    .ok_or_else(|| Error::Invariant("tensor transfer has no bound output".into()))
            })
            .collect()
    }
}

fn product_dtype(dtype: ProductDType) -> Result<crate::ffi::DType> {
    parse_dtype(dtype.storage_name())
}
