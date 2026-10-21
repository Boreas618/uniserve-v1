//! Admission of a batch's imports in descriptor order: product imports read
//! into the tensor store, latent trajectory versions read into the slot of
//! their one consumer, and cache imports, each supplied or resident transfer
//! copied into the units its install call names.

use std::collections::HashMap;
use std::sync::Arc;

use ipc::{
    Batch as BatchPlan, BufferId, CacheTransfer, CallKind, MediaCall, TensorExport, TensorRef,
    TransferHandle, TransferTransport,
};

use super::{Input, component_device, installs};
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::executor::transfer::latents::{self};
use crate::lane::Job;
use crate::memory::{Metadata, OutputPlacement, ValueRange};
use crate::{Error, Result};

/// The calls reading each supplied product, as an input or a predicate.
pub(super) fn consumers(plan: &BatchPlan) -> HashMap<&TensorRef, Vec<usize>> {
    let mut consumers: HashMap<&TensorRef, Vec<usize>> = HashMap::new();
    for (index, call) in plan.calls.iter().enumerate() {
        for product in call.tensor_inputs().chain(call.predicate.as_ref()) {
            let indices = consumers.entry(product).or_default();
            if indices.last() != Some(&index) {
                indices.push(index);
            }
        }
    }
    consumers
}

/// Whether a host media consumer reads `export` in place: a muxer always,
/// a video encoder a segment on this host.
pub(super) fn borrowed(
    backend: &WorkerBackend,
    plan: &BatchPlan,
    readers: &[usize],
    export: &TensorExport,
) -> bool {
    let shared = backend.worker_storage().is_some_and(|storage| {
        storage
            .transports
            .get(crate::transfer::TransportKind::PosixShm)
            .is_some()
    });
    let host = &backend.info().endpoint.host;
    readers.iter().any(|&index| {
        let call = &plan.calls[index];
        call.inputs.contains(&export.product)
            && (call.kind == CallKind::Media(MediaCall::Muxing)
                || (call.kind == CallKind::Media(MediaCall::VideoEncoding)
                    && shared
                    && !matches!(export.value, TransferHandle::Latent { .. })
                    && export.value.tensors().iter().any(|tensor| {
                        tensor.locations.iter().any(|location| {
                            matches!(location.transport, TransferTransport::PosixShm { .. })
                                && location.source.host == *host
                        })
                    })))
    })
}

/// The metadata an imported product carries.
fn metadata(handle: &TransferHandle) -> Result<Option<Metadata>> {
    Ok(match handle {
        TransferHandle::Encoder { height, width, .. } => Some(Metadata::Feature {
            height: *height,
            width: *width,
        }),
        TransferHandle::DeviceProduct {
            height,
            width,
            value_range,
            ..
        } if *height != 0 => Some(Metadata::Image {
            height: *height,
            width: *width,
            value_range: match value_range.as_str() {
                "signed_unit" => Some(ValueRange::SignedUnit),
                "unit" => Some(ValueRange::Unit),
                "" => None,
                other => {
                    return Err(Error::Invalid(format!(
                        "image product has an unknown value range {other:?}"
                    )));
                }
            },
        }),
        _ => None,
    })
}

/// Admit product imports from the first one not yet admitted, recording
/// their reads' handoff fences into `job`. Returns false while a refused
/// import waits for read credits.
pub(super) fn import_products(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    plan: &BatchPlan,
    job: &Job,
) -> Result<bool> {
    let start = batch.inputs().state.started;
    if start >= plan.input_products.len() {
        return Ok(true);
    }
    let readers = consumers(plan);
    let allocations: HashMap<BufferId, ipc::BufferAllocation> = plan
        .buffer_allocations
        .iter()
        .map(|allocation| (allocation.buffer, *allocation))
        .collect();

    for (index, export) in plan.input_products.iter().enumerate().skip(start) {
        batch.inputs_mut().state.started = index;
        let product = &export.product;
        let indices = readers.get(product).ok_or_else(|| {
            Error::Invalid("a supplied input product has no consumer in its batch".into())
        })?;
        if borrowed(backend, plan, indices, export) {
            batch
                .inputs_mut()
                .state
                .inputs
                .insert(product.buffer_id(), Input::Borrowed);
            continue;
        }
        if matches!(export.value, TransferHandle::Latent { .. }) {
            match import_latent(backend, plan, export, indices, job) {
                Ok(input) => {
                    batch
                        .inputs_mut()
                        .state
                        .inputs
                        .insert(product.buffer_id(), input);
                    continue;
                }
                Err(Error::ReadBackpressure { returns }) => {
                    let inputs = batch.inputs_mut();
                    inputs.backpressure = Some(returns);
                    inputs.state.awaiting_reads = true;
                    return Ok(false);
                }
                Err(error) => return Err(error),
            }
        }

        // A product may feed several calls, but it has one destination.
        let device = component_device(backend, &plan.calls[indices[0]].component)?;
        for &other in &indices[1..] {
            if component_device(backend, &plan.calls[other].component)? != device {
                return Err(Error::Invalid(
                    "transferred product requires one consumer device per batch".into(),
                ));
            }
        }
        let stream = backend.transfer_stream(plan.calls[indices[0]].kind, device)?;
        let tensor = export
            .value
            .tensors()
            .first()
            .ok_or_else(|| Error::Invalid("a supplied product carries no tensor".into()))?;
        let metadata = metadata(&export.value)?;
        let slots: HashMap<_, _> = backend
            .requests()
            .peek(product.request_key.request_id.0)
            .filter(|request| request.key() == product.request_key)
            .map(|request| (product.request_key, request.slot()))
            .into_iter()
            .collect();
        let storage = backend
            .worker_storage()
            .ok_or_else(|| Error::Unsupported("product imports require rank storage".into()))?;
        if storage.transports.is_empty() {
            return Err(Error::Unsupported(
                "cross-call input requires a configured transport".into(),
            ));
        }
        // The reads' handoff fences are recorded into the job of the lane
        // that owns the consumers' stream; the reads start once it ran.
        let imported = storage.tensors.import_tensor(
            product,
            tensor,
            device,
            &storage.transports,
            OutputPlacement {
                request_slots: Some(&slots),
                allocations: Some(&allocations),
                ..OutputPlacement::default()
            },
            metadata,
            job.on(stream.as_ref()),
        );
        match imported {
            Ok(read) => {
                batch
                    .inputs_mut()
                    .state
                    .inputs
                    .insert(product.buffer_id(), Input::Tensor { read, device });
            }
            Err(Error::ReadBackpressure { returns }) => {
                // The refusal started no read; the same product is admitted
                // again once credits return.
                let inputs = batch.inputs_mut();
                inputs.backpressure = Some(returns);
                inputs.state.awaiting_reads = true;
                return Ok(false);
            }
            Err(error) => return Err(error),
        }
    }
    let inputs = batch.inputs_mut();
    inputs.state.started = plan.input_products.len();
    inputs.state.awaiting_reads = false;
    inputs.backpressure = None;
    Ok(true)
}

/// Read a latent trajectory version into the slot of its one consumer,
/// whose scheduler parameters name the pages and the version. The reads'
/// handoff fences are recorded into `job`.
fn import_latent(
    backend: &mut WorkerBackend,
    plan: &BatchPlan,
    export: &TensorExport,
    readers: &[usize],
    job: &Job,
) -> Result<Input> {
    let [index] = readers else {
        return Err(Error::Invalid(
            "latent transfer must have one consumer".into(),
        ));
    };
    let call = &plan.calls[*index];
    let params = plan
        .latent_params
        .iter()
        .find(|params| params.request_key == call.request_key && params.call_id == call.call_id)
        .ok_or_else(|| Error::Invalid("latent transfer has no scheduler parameters".into()))?;
    let device = component_device(backend, &call.component)?;
    let stream = backend.transfer_stream(call.kind, device)?;
    let slot = backend
        .requests()
        .peek(export.product.request_key.request_id.0)
        .filter(|request| request.key() == export.product.request_key)
        .map(|request| request.slot())
        .ok_or_else(|| Error::Invalid("latent transfer has no request slot".into()))?;
    let parts = backend.parts();
    let storage = parts
        .storage
        .ok_or_else(|| Error::Unsupported("latent imports require rank storage".into()))?;
    let mut image = parts
        .image
        .ok_or_else(|| Error::Invalid("latent transfer requires physical latent storage".into()))?;
    let (pool, latent) = image.latent_pool()?;
    let input = latents::import(
        pool,
        latent,
        &storage.transports,
        export,
        params,
        slot,
        job.on(stream.as_ref()),
    )?;
    Ok(Input::Latent { input, device })
}

/// Reserve every cache import: each supplied or resident transfer is copied
/// into the pages of its one install call's slot. An accepted import is
/// recorded at once, so a later refusal abandons it with the batch.
pub(super) fn import_cache(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    plan: &BatchPlan,
) -> Result<()> {
    let resident = batch.inputs().resident.clone();
    if plan.cache_inputs.is_empty() && resident.is_empty() {
        return Ok(());
    }
    let mut consumers: HashMap<BufferId, usize> = HashMap::new();
    for call in installs(plan) {
        if let Some(source) = call.cache_input {
            *consumers.entry(source).or_default() += 1;
        }
    }
    let transfers: Vec<Arc<CacheTransfer>> = plan
        .cache_inputs
        .iter()
        .cloned()
        .map(Arc::new)
        .chain(resident)
        .collect();
    if transfers
        .iter()
        .any(|transfer| consumers.get(&transfer.source) != Some(&1))
    {
        return Err(Error::Invalid(
            "cache input requires one installation consumer".into(),
        ));
    }

    let parts = backend.parts();
    let storage = parts
        .storage
        .ok_or_else(|| Error::Unsupported("cache input requires rank storage".into()))?;
    if storage.transports.is_empty() {
        return Err(Error::Unsupported(
            "cross-call input requires a configured transport".into(),
        ));
    }
    let transports = Arc::clone(&storage.transports);
    let prefix_cache = storage
        .prefix_cache
        .as_mut()
        .ok_or_else(|| Error::Invalid("cache input requires physical cache storage".into()))?;
    for transfer in transfers {
        let source = transfer.source;
        // Starts were installed before preparation; the request pool is the
        // sole source of the consumer's slot.
        let slot = parts.requests.get(source.owner.request_id.0)?.slot() as u32;
        let tables = prefix_cache.tables().for_batch(slot, &plan.block_tables)?;
        let initialized = plan
            .new_cache_units
            .iter()
            .filter(|allocation| allocation.request_pool_idx == slot)
            .flat_map(|allocation| allocation.unit_ids.iter().map(|unit| unit.0))
            .collect();
        let import = prefix_cache.reserve_import(
            transfer,
            slot,
            tables,
            initialized,
            Arc::clone(&transports),
        )?;
        batch
            .inputs_mut()
            .state
            .inputs
            .insert(source, Input::Cache(import));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_products_carry_their_value_range() -> Result<()> {
        let tensor = ipc::TensorTransfer {
            shape: vec![1],
            locations: Vec::new(),
        };
        let image = TransferHandle::DeviceProduct {
            height: 4,
            width: 2,
            value_range: "signed_unit".into(),
            tensor: tensor.clone(),
        };
        assert_eq!(
            metadata(&image)?,
            Some(Metadata::Image {
                height: 4,
                width: 2,
                value_range: Some(ValueRange::SignedUnit),
            })
        );
        let plain = TransferHandle::DeviceProduct {
            height: 0,
            width: 0,
            value_range: String::new(),
            tensor,
        };
        assert_eq!(metadata(&plain)?, None);
        Ok(())
    }
}
