//! Binding a batch's outputs and writing products.

use super::*;
use crate::ffi::MemoryKind;
use crate::memory::allocation::StorageView;
use crate::stream::Issue;
use crate::transfer::{Spans, copy_planner};
use crate::{Error, Result};
use cuda::SourceOrder;
use ipc::{BufferAllocation, BufferId, RequestKey, TensorRef};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

/// Scheduler placement of a group of outputs. Maps are keyed by buffer id.
#[derive(Clone, Copy, Default)]
pub struct OutputPlacement<'a> {
    /// Request slots of the owning requests, for relay outputs.
    pub request_slots: Option<&'a HashMap<RequestKey, usize>>,
    /// Buffer-pool allocations of persistent outputs.
    pub allocations: Option<&'a HashMap<BufferId, BufferAllocation>>,
    /// The logical region this rank stores of a sharded output.
    pub regions: Option<&'a HashMap<BufferId, Vec<Range<u64>>>>,
    /// Actual logical shapes overriding the reference's bound.
    pub shapes: Option<&'a HashMap<BufferId, Vec<u64>>>,
}

/// Production ends a numerical write; commit makes its value discoverable.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriteState {
    Reserved,
    Deferred,
    Produced,
    Committed,
}

impl WriteState {
    pub(crate) fn produced(self) -> bool {
        matches!(self, Self::Produced | Self::Committed)
    }
}

impl ProductStore {
    /// Bind each group of outputs atomically; a failure releases every group
    /// bound by this call.
    pub fn bind_output_groups(
        &self,
        groups: &[Vec<(TensorRef, MemoryKind)>],
        placement: OutputPlacement<'_>,
    ) -> Result<Vec<Vec<ProductBuffer>>> {
        let mut state = lock(&self.state);
        let mut outputs = Vec::new();
        let mut acquired = Vec::new();
        for group in groups.iter().filter(|group| !group.is_empty()) {
            match self.bind_group(&mut state, group, placement) {
                Ok(buffers) => {
                    acquired.extend(buffers.iter().cloned());
                    outputs.push(buffers);
                }
                Err(error) => {
                    self.rollback(&mut state, acquired)?;
                    return Err(error);
                }
            }
        }
        Ok(outputs)
    }

    /// Bind one group of outputs: persistent outputs into their pool
    /// allocations, or scalar outputs into relay entries.
    pub fn bind_outputs(
        &self,
        bindings: &[(TensorRef, MemoryKind)],
        placement: OutputPlacement<'_>,
    ) -> Result<Vec<ProductBuffer>> {
        let mut state = lock(&self.state);
        self.bind_group(&mut state, bindings, placement)
    }

    /// Reserve encoder features in their pool allocations.
    pub(crate) fn reserve_features(
        &self,
        bindings: &[(TensorRef, MemoryKind)],
        allocations: &HashMap<BufferId, BufferAllocation>,
        placement: OutputPlacement<'_>,
    ) -> Result<Vec<ProductBuffer>> {
        let mut state = lock(&self.state);
        self.bind_persistent(&mut state, bindings, allocations, placement, true)
    }

    pub(super) fn logical_shape(reference: &TensorRef, placement: OutputPlacement<'_>) -> Vec<u64> {
        placement
            .shapes
            .and_then(|shapes| shapes.get(&reference.buffer_id()))
            .cloned()
            .unwrap_or_else(|| device_shape(reference))
    }

    pub(super) fn bind_group(
        &self,
        state: &mut State,
        bindings: &[(TensorRef, MemoryKind)],
        placement: OutputPlacement<'_>,
    ) -> Result<Vec<ProductBuffer>> {
        if bindings.is_empty() {
            return Ok(Vec::new());
        }
        let mut persistent = 0;
        for (reference, _) in bindings {
            let shape = Self::logical_shape(reference, placement);
            if !reference.shape_bound.contains_shape(&shape) {
                return Err(Error::Invalid(
                    "tensor binding shape disagrees with its logical bounds".into(),
                ));
            }
            if let Some(region) = placement
                .regions
                .and_then(|regions| regions.get(&reference.buffer_id()))
                && !within(region, &shape)
            {
                return Err(Error::Invalid(
                    "tensor binding region disagrees with its logical bounds".into(),
                ));
            }
            if placement
                .allocations
                .is_some_and(|allocations| allocations.contains_key(&reference.buffer_id()))
            {
                persistent += 1;
            }
        }
        if persistent > 0 {
            if persistent != bindings.len() {
                return Err(Error::Invalid(
                    "persistent-buffer bindings cannot share a generic group".into(),
                ));
            }
            let allocations = placement.allocations.ok_or_else(|| {
                Error::Invariant("persistent buffers lost their placement".into())
            })?;
            return self.bind_persistent(state, bindings, allocations, placement, false);
        }
        if let Some(request_slots) = placement.request_slots {
            return self.bind_relays(state, bindings, request_slots);
        }
        Err(Error::Invalid(
            "tensor output requires a buffer allocation or request relay slot".into(),
        ))
    }

    pub(super) fn bind_persistent(
        &self,
        state: &mut State,
        bindings: &[(TensorRef, MemoryKind)],
        allocations: &HashMap<BufferId, BufferAllocation>,
        placement: OutputPlacement<'_>,
        feature: bool,
    ) -> Result<Vec<ProductBuffer>> {
        self.reclaim(state)?;
        let mut native = Vec::new();
        let mut candidates = Vec::new();
        for (reference, device) in bindings {
            let id = reference.buffer_id();
            let allocation = *allocations.get(&id).ok_or_else(|| {
                Error::Invalid("persistent output has no buffer allocation".into())
            })?;
            if feature {
                if !matches!(
                    reference.dtype,
                    ipc::DType::F16 | ipc::DType::BF16 | ipc::DType::F32
                ) {
                    return Err(Error::Invalid(
                        "encoder feature dtype is unsupported".into(),
                    ));
                }
                let bytes = reference.max_bytes();
                if bytes > self.max_feature_bytes as u64 {
                    return Err(Error::Resource(
                        "encoder feature exceeds the fixed feature byte capacity",
                    ));
                }
                if !self.devices.contains(device) {
                    return Err(Error::Invalid(
                        "encoder feature names an undeclared device".into(),
                    ));
                }
            }
            native.push(id);
            candidates.push((reference, *device, allocation));
        }
        state.validate_bindings(&native)?;

        let mut buffers = Vec::new();
        let result = (|| -> Result<()> {
            for (reference, device, allocation) in candidates {
                let dtype = product_dtype(reference.dtype)?;
                let shape = Self::logical_shape(reference, placement);
                let region = placement
                    .regions
                    .and_then(|regions| regions.get(&reference.buffer_id()))
                    .filter(|region| **region != full(&shape))
                    .cloned();
                let local = region
                    .as_deref()
                    .map_or_else(|| shape.clone(), region_shape);
                // An allocation covering the whole logical tensor stores it at
                // logical coordinates, so imports can widen the shard later.
                let full_storage =
                    region.is_some() && allocation.bytes >= shape_bytes(&shape, dtype)?;
                let storage_shape = if full_storage { &shape } else { &local };

                let (binding, arena) = {
                    let mut pool = lock(&self.buffer_pool);
                    let (binding, arena) = pool.bind(
                        allocation,
                        device,
                        shape_bytes(storage_shape, dtype)?,
                        dtype.bytes() as u64,
                    )?;
                    (binding, Arc::clone(arena))
                };
                let views = (|| -> Result<(StorageView, StorageView)> {
                    let storage = StorageView::contiguous(
                        arena as Arc<dyn crate::ffi::Backing>,
                        binding.physical_offset() as usize,
                        dtype,
                        &signed(storage_shape),
                    )?;
                    let view = match (&region, full_storage) {
                        (Some(region), true) => storage.region(region)?,
                        _ => storage.clone(),
                    };
                    Ok((view, storage))
                })();
                let (view, storage) = match views {
                    Ok(views) => views,
                    Err(error) => {
                        lock(&self.buffer_pool).release(&binding)?;
                        return Err(error);
                    }
                };
                let buffer = create_buffer(
                    reference,
                    device,
                    view,
                    storage,
                    shape,
                    region,
                    feature,
                    Backing::Persistent(binding),
                );
                state
                    .buffers
                    .insert(buffer_key(reference.buffer_id()), buffer.clone());
                buffers.push(buffer);
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.rollback(state, buffers)?;
            return Err(error);
        }
        Ok(buffers)
    }

    pub(super) fn rollback(&self, state: &mut State, buffers: Vec<ProductBuffer>) -> Result<()> {
        for buffer in buffers.into_iter().rev() {
            state.buffers.remove(&buffer_key(buffer.id()));
            self.release_backing(state, &buffer)?;
        }
        Ok(())
    }

    /// Publish a value the producer wrote into the bound view: `shape` is
    /// the value's actual shape, fenced behind what the producer `stream`
    /// issued.
    pub fn produced(
        &self,
        buffer: &ProductBuffer,
        shape: &[u64],
        stream: impl Issue,
        metadata: Option<Metadata>,
    ) -> Result<StorageView> {
        let state = lock(&self.state);
        state.require_buffer(buffer)?;
        let view = buffer.view();
        let value = if buffer.region().is_some() {
            view
        } else {
            value_view(&view, shape, elements(shape)?)?
        };
        self.write_value(buffer, &value, stream, metadata)
    }

    /// Copy `value` into the product's bound view (no copy when `value` is
    /// that view) in the producer `stream`'s order, then fence it.
    pub(crate) fn write(
        &self,
        buffer: &ProductBuffer,
        value: &StorageView,
        stream: impl Issue,
        metadata: Option<Metadata>,
    ) -> Result<StorageView> {
        let state = lock(&self.state);
        state.require_buffer(buffer)?;
        self.write_value(buffer, value, stream, metadata)
    }

    pub(super) fn write_value(
        &self,
        buffer: &ProductBuffer,
        value: &StorageView,
        stream: impl Issue,
        metadata: Option<Metadata>,
    ) -> Result<StorageView> {
        if let Some(metadata) = metadata {
            metadata.validate()?;
        }
        let feature = {
            let state = lock(buffer);
            if state.state.produced() {
                return Err(Error::Invariant(
                    "device product was published more than once".into(),
                ));
            }
            state.feature
        };
        let (reference, target, region) = {
            let views = buffer.views();
            (
                views.reference.clone(),
                views.view.clone(),
                views.region.clone(),
            )
        };
        match metadata {
            Some(Metadata::Feature { .. }) if !feature => {
                return Err(Error::Invalid(
                    "feature metadata requires feature admission".into(),
                ));
            }
            Some(Metadata::Feature { .. }) => {}
            _ if feature => {
                return Err(Error::Invalid(
                    "encoder feature requires spatial metadata".into(),
                ));
            }
            _ => {}
        }
        // Producers write the store's dtype: a feature's cast belongs to the
        // producing computation.
        if value.dtype() != target.dtype() {
            return Err(Error::Invalid(
                "tensor export changes its declared dtype".into(),
            ));
        }
        let shape: Vec<u64> = value.shape().iter().map(|&extent| extent as u64).collect();
        let matches = match &region {
            Some(region) => shape == region_shape(region),
            None => reference.shape_bound.contains_shape(&shape),
        };
        if !matches {
            return Err(Error::Invalid(format!(
                "tensor export of shape {shape:?} changes the declared shape of output {} of {:?}",
                reference.output_index, reference.producer_call_id
            )));
        }
        let extent = value.numel();
        if extent > target.numel() {
            return Err(Error::Invariant(
                "device product exceeds its registered shape bound".into(),
            ));
        }
        let view = match region {
            Some(_) => target,
            None => value_view(&target, &shape, extent)?,
        };

        // An exact alias needs no copy; noncontiguous shard strides stay intact.
        if view.address() != value.address() || view.strides() != value.strides() {
            let destination = Spans::single(view.clone())?;
            let source = Spans::single(value.clone())?;
            match (view.device(), value.device()) {
                (MemoryKind::Cuda(device), _) | (_, MemoryKind::Cuda(device)) => {
                    if stream.stream().is_none() {
                        return Err(Error::Invalid(
                            "a device product write requires its producer stream".into(),
                        ));
                    }
                    let mut plan = copy_planner(Some(device), SourceOrder::Stream)?;
                    destination.plan_copy(&source, &mut plan)?;
                    // SAFETY: the destination is reserved for this product and
                    // the producer keeps its value until its stream passes.
                    unsafe { stream.copy_strided(&plan) }?;
                }
                _ => {
                    let mut plan = copy_planner(None, SourceOrder::Stream)?;
                    destination.plan_copy(&source, &mut plan)?;
                    // SAFETY: host operands, written in the producer's order;
                    // the destination has no reader yet.
                    unsafe { stream.host().copy_strided(&plan) }?;
                }
            }
        }

        let fence = self.producer_fence(view.device(), stream, 1)?;
        lock(buffer).produced(
            shape.iter().map(|&extent| extent as usize).collect(),
            extent as usize,
            fence,
        );
        buffer.views().metadata = metadata;
        Ok(view)
    }

    /// Defer a write to a later call.
    pub fn defer_write(&self, write: &ProductBuffer) -> Result<()> {
        let state = lock(&self.state);
        state.require_buffer(write)?;
        lock(write).defer()
    }

    pub(crate) fn validate_writes(&self, writes: &[ProductBuffer]) -> Result<()> {
        lock(&self.state).validate_writes(writes)
    }

    /// Make produced writes discoverable; deferred writes stay pending.
    pub fn commit_writes(&self, writes: &[ProductBuffer]) -> Result<()> {
        lock(&self.state).commit_writes(writes)
    }

    /// Release the storage of writes whose call failed.
    pub fn abandon_writes(&self, writes: &[ProductBuffer]) -> Result<()> {
        let mut state = lock(&self.state);
        state.abandon_writes(writes);
        self.reclaim(&mut state)
    }
}
