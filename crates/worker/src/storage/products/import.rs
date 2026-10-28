//! Products imported from other ranks and workers.

use super::*;
use crate::ffi::{DType, MemoryKind};
use crate::storage::allocation::StorageView;
use crate::stream::Issue;
use crate::transfer::{ReadRef, Spans, Transports, fetch};
use crate::{Error, Result};
use ipc::{TensorRef, TensorTransfer, subtract_region};
use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::sync::{Arc, Mutex};

pub(super) struct StoredImport {
    pub(super) state: Mutex<TensorImport>,
    pub(super) destination: StorageView,
    pub(super) metadata: Option<Metadata>,
}

/// Missing coverage of one product materialized once and shared by its
/// concurrent readers.
#[derive(Clone)]
pub(super) struct ImportHandle(pub(super) Arc<StoredImport>);

impl Deref for ImportHandle {
    type Target = Mutex<TensorImport>;

    fn deref(&self) -> &Self::Target {
        &self.0.state
    }
}

/// Revive `buffer`, a copy of an imported product this rank released, for
/// a later reader; true when it holds the product's value for that reader.
///
/// A copy released after it received its value still holds it: its storage
/// returns only once a reclaim removes it, and its pending fences order
/// only accesses already issued. Reviving it registers it under its
/// producing call again, so its owner's release retires it once more. A
/// copy released before its value arrived (an abandoned import, whose reads
/// were cancelled) cannot serve and stays released.
fn revive(state: &mut State, buffer: &ProductBuffer) -> bool {
    let mut value = lock(buffer);
    if !value.released {
        return true;
    }
    if !value.state.produced() {
        return false;
    }
    value.released = false;
    let id = value.id;
    drop(value);
    state
        .calls
        .entry((id.owner, id.producer_call_id))
        .or_default()
        .insert(buffer_key(id));
    true
}

pub(super) fn import_read(imported: &ImportHandle) -> Arc<ProductRead> {
    let buffer = lock(imported).buffer.clone();
    Arc::new(ProductRead {
        value: imported.0.destination.clone(),
        region: None,
        metadata: imported.0.metadata,
        lease: Mutex::new(TensorRead {
            buffer,
            consumer: None,
            imported: Some(imported.clone()),
            complete: false,
        }),
    })
}

/// Missing coverage materialized once and shared by its concurrent readers.
pub(crate) struct TensorImport {
    pub buffer: ProductBuffer,
    pub tickets: Vec<ReadRef>,
    pub users: usize,
    pub committed: bool,
}

impl TensorImport {
    pub(crate) fn new(buffer: ProductBuffer, tickets: Vec<ReadRef>, committed: bool) -> Self {
        Self {
            buffer,
            tickets,
            users: 1,
            committed,
        }
    }

    pub(crate) fn share(&mut self) {
        lock(&self.buffer).readers += 1;
        self.users += 1;
    }
}

impl ProductStore {
    /// Materialize `tensor` (a product published on other ranks) on
    /// `device`: share a pending import of the same product, widen a
    /// committed shard by its missing regions, or reserve storage and read
    /// the whole tensor. Reads are planned against the whole destination and
    /// submitted as one reservation, ordered after `stream`. A failed
    /// submission retains any in-flight destination.
    #[expect(clippy::too_many_arguments, reason = "one import's complete inputs")]
    pub fn import_tensor(
        &self,
        reference: &TensorRef,
        tensor: &TensorTransfer,
        device: MemoryKind,
        transports: &Transports,
        placement: OutputPlacement<'_>,
        metadata: Option<Metadata>,
        stream: impl Issue,
    ) -> Result<Arc<ProductRead>> {
        let id = reference.buffer_id();
        let key = buffer_key(id);
        let shape = tensor.shape.clone();
        let mut state = lock(&self.state);
        // A copy of the product this rank released (a consumed predicate)
        // stays indexed until a reclaim finds its accesses retired, and
        // nothing reclaims between that release and the next call reading
        // the product. Reclaim first, as every binding does: a retired copy
        // leaves and the product is read anew, and a copy still draining is
        // revived (`revive`).
        self.reclaim(&mut state)?;

        if let Some(pending) = state.imports.get(&key).cloned() {
            let mut value = lock(&pending);
            let buffer = value.buffer.clone();
            if buffer.views().reference != *reference
                || pending.0.destination.device() != device
                || pending.0.destination.shape() != signed(&shape).as_slice()
                || pending.0.metadata != metadata
            {
                return Err(Error::Invalid(
                    "product import conflicts with pending materialization".into(),
                ));
            }
            if !revive(&mut state, &buffer) {
                return Err(Error::Invalid(
                    "product import requires a live published generation".into(),
                ));
            }
            value.share();
            drop(value);
            return Ok(import_read(&pending));
        }

        let existing = state
            .buffers
            .get(&key)
            .filter(|buffer| lock(buffer).state == WriteState::Committed)
            .cloned();
        let all = full(&shape);
        let (buffer, destination, missing) = match existing {
            Some(_) => {
                let buffer = Self::require_reference(&state, reference)?;
                // A committed copy is produced, so it always revives.
                revive(&mut state, &buffer);
                let value = lock(&buffer);
                let views = buffer.views();
                if device != value.device || views.metadata != metadata {
                    return Err(Error::Invalid(
                        "product import conflicts with resident ownership".into(),
                    ));
                }
                let (destination, missing) = match &views.region {
                    None => {
                        if value
                            .value_shape
                            .iter()
                            .map(|&extent| extent as u64)
                            .ne(shape.iter().copied())
                        {
                            return Err(Error::Invalid(
                                "product import changes resident tensor shape".into(),
                            ));
                        }
                        (
                            value_view(&views.view, &shape, value.extent as u64)?,
                            Vec::new(),
                        )
                    }
                    Some(region) => {
                        if !matches!(value.backing, Some(Backing::Persistent(_)))
                            || views.storage.shape() != signed(&shape).as_slice()
                        {
                            return Err(Error::Invalid(
                                "product import exceeds its reserved logical storage".into(),
                            ));
                        }
                        if value.transfers.iter().any(|read| !read.succeeded()) {
                            return Err(Error::Resource(
                                "product storage has pending physical reads",
                            ));
                        }
                        let mut missing = Vec::new();
                        subtract_region(&all, region, &mut missing);
                        (views.storage.clone(), missing)
                    }
                };
                drop(views);
                drop(value);
                (buffer, destination, missing)
            }
            None => {
                let shapes = HashMap::from([(id, shape.clone())]);
                let placement = OutputPlacement {
                    shapes: Some(&shapes),
                    ..placement
                };
                let bound = [(reference.clone(), device)];
                let buffers = if matches!(metadata, Some(Metadata::Feature { .. })) {
                    let allocations = placement.allocations.ok_or_else(|| {
                        Error::Invalid("persistent output has no buffer allocation".into())
                    })?;
                    self.bind_persistent(&mut state, &bound, allocations, placement, true)?
                } else {
                    self.bind_group(&mut state, &bound, placement)?
                };
                let buffer = buffers.into_iter().next().ok_or_else(|| {
                    Error::Invariant("import destination was not reserved".into())
                })?;
                let destination = value_view(&buffer.view(), &shape, elements(&shape)?)?;
                (buffer, destination, vec![all])
            }
        };

        let source_dtype = tensor
            .locations
            .first()
            .ok_or_else(|| Error::Invalid("tensor transfer has no locations".into()))
            .and_then(|location| DType::parse(&location.dtype))?;
        if destination.dtype() != source_dtype {
            if existing.is_none() {
                lock(&buffer).released = true;
                self.reclaim(&mut state)?;
            }
            return Err(Error::Invalid(
                "product import changes resident tensor dtype".into(),
            ));
        }

        lock(&buffer).readers += 1;
        let mut tickets = Vec::new();
        let submitted = (|| -> Result<()> {
            let mut locators = vec![None; tensor.locations.len()];
            let mut reads = Vec::new();
            for region in &missing {
                let target = Spans::single(destination.region(region)?)?;
                reads.extend(fetch::plan(
                    tensor,
                    &target,
                    transports,
                    region,
                    &mut locators,
                )?);
            }
            fetch::submit(&reads, stream, |read| {
                tickets.push(ReadRef::new(read));
                Ok(())
            })?;
            Ok(())
        })();
        let reads: Vec<_> = tickets
            .iter()
            .map(|ticket| Arc::clone(ticket.read()))
            .collect();
        lock(&buffer)
            .transfers
            .extend(tickets.iter().map(|ticket| ReadRef::new(ticket.read())));
        if let Err(error) = submitted {
            {
                let mut value = lock(&buffer);
                value.readers -= 1;
                if existing.is_none() {
                    value.released = true;
                }
            }
            drop(state);
            for read in reads {
                read.close()?;
            }
            let mut state = lock(&self.state);
            self.reclaim(&mut state)?;
            return Err(error);
        }

        let committed = existing.is_some() && buffer.region().is_none();
        let imported = ImportHandle(Arc::new(StoredImport {
            state: Mutex::new(TensorImport::new(buffer, tickets, committed)),
            destination,
            metadata,
        }));
        let read = import_read(&imported);
        state.imports.insert(key, imported);
        Ok(read)
    }

    pub(super) fn imported(read: &ProductRead) -> Result<ImportHandle> {
        let lease = lock(&read.lease);
        if lease.complete {
            return Err(Error::Invariant(
                "closed or ordinary tensor read is not an import".into(),
            ));
        }
        lease.imported.clone().ok_or_else(|| {
            Error::Invariant("closed or ordinary tensor read is not an import".into())
        })
    }

    pub(super) fn wait_imported(
        state: &State,
        read: &ProductRead,
        stream: impl Issue,
    ) -> Result<()> {
        let imported = Self::imported(read)?;
        let imported = lock(&imported);
        state.require_buffer(&imported.buffer)?;
        for ticket in &imported.tickets {
            ticket.read().result(stream)?;
        }
        let buffer = lock(&imported.buffer);
        Self::wait_producer(&buffer, stream, &mut HashSet::new())
    }

    /// Publish an import's materialized value on `stream`: a new product is
    /// produced and committed; a widened shard gets one fence over the
    /// union of its regions and becomes the whole logical tensor.
    pub fn complete_import(&self, read: &ProductRead, stream: impl Issue) -> Result<()> {
        let mut state = lock(&self.state);
        Self::wait_imported(&state, read, stream)?;
        let imported = Self::imported(read)?;
        let (buffer, committed) = {
            let value = lock(&imported);
            (value.buffer.clone(), value.committed)
        };
        if committed {
            return Ok(());
        }
        let destination = imported.0.destination.clone();
        if !lock(&buffer).state.produced() {
            self.write_value(&buffer, &destination, stream, imported.0.metadata)?;
            state.commit_writes(std::slice::from_ref(&buffer))?;
        } else {
            // Existing leases retain their original shard views. A new fence
            // covers the union, including every newly fetched region.
            let fence = self.producer_fence(destination.device(), stream, 1)?;
            let previous = lock(&buffer)
                .producer
                .as_ref()
                .map(|fence| Arc::clone(&fence.event));
            if let Some(previous) = previous {
                let owner = buffer.clone();
                stream.defer(
                    &self.events,
                    vec![previous],
                    Box::new(move || {
                        drop(owner);
                        Ok(())
                    }),
                )?;
            }
            {
                let mut views = buffer.views();
                views.view = destination.clone();
                views.region = None;
            }
            let mut value = lock(&buffer);
            value.value_shape = destination
                .shape()
                .iter()
                .map(|&extent| extent as usize)
                .collect();
            value.extent = destination.numel() as usize;
            value.producer = fence;
        }
        lock(&imported).committed = true;
        Ok(())
    }
}
