//! Read leases of committed products.

use super::*;
use crate::ffi::Device;
use crate::storage::memory::{StorageView, device_name};
use crate::stream::Issue;
use crate::{Error, Result};
use std::collections::{HashMap, HashSet};
use std::ops::{Deref, Range};
use std::sync::{Arc, Mutex};
use uniserve_core::CallId;
use uniserve_cuda::{Event, Stream};
use uniserve_ipc::TensorRef;

/// One consumer's lease of a product. The value view stays stable even when
/// another import widens the product's resident coverage.
pub struct ProductRead {
    pub(super) value: StorageView,
    pub(super) region: Option<Vec<Range<u64>>>,
    pub(super) metadata: Option<Metadata>,
    pub(super) lease: Mutex<TensorRead<ProductBuffer, ImportHandle>>,
}

impl ProductRead {
    pub fn value(&self) -> &StorageView {
        &self.value
    }

    pub fn region(&self) -> Option<&[Range<u64>]> {
        self.region.as_deref()
    }

    pub fn metadata(&self) -> Option<Metadata> {
        self.metadata
    }

    pub fn complete(&self) -> bool {
        lock(&self.lease).complete
    }

    /// The transport reads materializing an import (empty for an ordinary
    /// lease); callers wait for them before `ProductStore::wait_import`.
    pub fn reads(&self) -> Vec<Arc<crate::transfer::Read>> {
        let imported = lock(&self.lease).imported.clone();
        imported.map_or_else(Vec::new, |imported| {
            lock(&imported)
                .tickets
                .iter()
                .map(|ticket| Arc::clone(ticket.read()))
                .collect()
        })
    }
}

/// One consumer's access. The consumer keeps a stable tensor view even when
/// another import extends the buffer's resident coverage.
pub(crate) struct TensorRead<B, I> {
    pub buffer: B,
    pub consumer: Option<CallId>,
    pub imported: Option<I>,
    pub complete: bool,
}

impl<B, I> TensorRead<B, I> {
    pub(crate) fn complete<T, C>(&mut self) -> Option<I>
    where
        B: Deref<Target = Mutex<Buffer<T, C>>>,
        I: Deref<Target = Mutex<TensorImport<B, T>>>,
    {
        if self.complete {
            return None;
        }
        self.complete = true;
        lock(&self.buffer).readers -= 1;

        let imported = self.imported.take()?;
        let mut import = lock(&imported);
        import.users -= 1;
        let last = import.users == 0;
        if last && !lock(&self.buffer).state.produced() {
            lock(&self.buffer).released = true;
        }
        drop(import);
        last.then_some(imported)
    }
}

impl ProductStore {
    /// Lease committed products for consumer calls. The whole batch is
    /// resolved before any lease is taken. `stream` (required for device
    /// products) waits on each producer fence once.
    pub fn consume_batch(
        &self,
        requests: &[(TensorRef, CallId, Option<Device>)],
        device: Option<Device>,
        stream: impl Issue,
    ) -> Result<Vec<Arc<ProductRead>>> {
        let state = lock(&self.state);
        let mut resolved = Vec::with_capacity(requests.len());
        let mut waited = HashSet::new();
        for (reference, consumer, requested) in requests {
            let buffer = Self::require_reference(&state, reference)?;
            let value = lock(&buffer);
            if value.released {
                return Err(Error::Invalid(
                    "device product was consumed after logical release".into(),
                ));
            }
            if let (Some(shared), Some(requested)) = (device, requested)
                && shared != *requested
            {
                return Err(Error::Invalid(
                    "device-product batch names conflicting consumer devices".into(),
                ));
            }
            if let Some(target) = device.or(*requested)
                && device_name(target) != value.device
            {
                return Err(Error::Invalid(
                    "device product consumer names a different device".into(),
                ));
            }
            Self::wait_producer(&value, stream, &mut waited)?;
            let shape: Vec<u64> = value
                .value_shape
                .iter()
                .map(|&extent| extent as u64)
                .collect();
            let extent = value.extent as u64;
            drop(value);
            let views = buffer.views();
            let read = Arc::new(ProductRead {
                value: value_view(&views.view, &shape, extent)?,
                region: views.region.clone(),
                metadata: views.metadata,
                lease: Mutex::new(TensorRead {
                    buffer: buffer.clone(),
                    consumer: Some(*consumer),
                    imported: None,
                    complete: false,
                }),
            });
            drop(views);
            resolved.push((buffer, read));
        }
        for (buffer, _) in &resolved {
            lock(buffer).readers += 1;
        }
        Ok(resolved.into_iter().map(|(_, read)| read).collect())
    }

    /// Lease one committed product.
    pub fn consume(
        &self,
        reference: &TensorRef,
        consumer: CallId,
        device: Option<Device>,
        stream: impl Issue,
    ) -> Result<Arc<ProductRead>> {
        let mut reads =
            self.consume_batch(&[(reference.clone(), consumer, device)], None, stream)?;
        reads
            .pop()
            .ok_or_else(|| Error::Invariant("a consumed product produced no lease".into()))
    }

    /// End leases after fencing their consuming `stream`. A consumer's own
    /// output fence on that stream (`after_writes`) serves as the reader
    /// fence; otherwise one is recorded.
    pub fn complete_reads(
        &self,
        reads: &[Arc<ProductRead>],
        stream: impl Issue,
        after_writes: &[ProductBuffer],
    ) -> Result<()> {
        let mut state = lock(&self.state);
        let mut seen = HashSet::new();
        let pending: Vec<_> = reads
            .iter()
            .filter(|read| !read.complete() && seen.insert(Arc::as_ptr(read) as usize))
            .collect();
        if pending.is_empty() {
            return Ok(());
        }
        for read in &pending {
            let buffer = lock(&read.lease).buffer.clone();
            state.require_buffer(&buffer)?;
            if let (Some(consumer), Device::Cuda(device)) = (stream.stream(), read.value.device())
                && consumer.device().0 != device
            {
                return Err(Error::Invariant(
                    "device-product reader completed on a different device".into(),
                ));
            }
        }

        let mut write_fences = HashMap::new();
        for write in after_writes {
            state.require_buffer(write)?;
            let value = lock(write);
            if let Some(fence) = &value.producer {
                write_fences
                    .entry((
                        value.id.producer_call_id,
                        value.device.clone(),
                        fence.stream,
                    ))
                    .or_insert_with(|| Arc::clone(&fence.event));
            }
        }

        let mut recorded = HashMap::<i32, Arc<Event>>::new();
        for read in &pending {
            let Device::Cuda(device) = read.value.device() else {
                continue;
            };
            let raw = stream.stream().map(Stream::raw).ok_or_else(|| {
                Error::Invalid("device product reads complete on their consumer stream".into())
            })?;
            let (buffer, consumer) = {
                let lease = lock(&read.lease);
                (lease.buffer.clone(), lease.consumer)
            };
            let name = lock(&buffer).device.clone();
            let event = consumer
                .and_then(|consumer| write_fences.get(&(consumer, name, raw)))
                .cloned();
            let event = match event {
                Some(event) => event,
                None => match recorded.get(&device) {
                    Some(event) => Arc::clone(event),
                    None => {
                        let event = self.events.acquire(device, false, false)?;
                        self.events.record(&event, stream)?;
                        recorded.insert(device, Arc::clone(&event));
                        event
                    }
                },
            };
            if lock(&buffer).retain_reader(&event) {
                self.events.retain(&event, device, 1)?;
            }
        }

        let mut closing = Vec::new();
        let mut released = Vec::new();
        for read in pending {
            let (buffer, completed) = {
                let mut lease = lock(&read.lease);
                let buffer = lease.buffer.clone();
                (buffer, lease.complete())
            };
            if let Some(imported) = completed {
                state.imports.remove(&buffer_key(buffer.id()));
                let import = lock(&imported);
                closing.extend(
                    import
                        .tickets
                        .iter()
                        .map(|ticket| (Arc::clone(ticket.read()), !import.committed)),
                );
            }
            if lock(&buffer).released {
                released.push(buffer);
            }
        }
        drop(state);

        // Read observers may reenter the store; storage stays retained by
        // its native dependencies while they run.
        for (read, cancel) in closing {
            if cancel {
                read.cancel()?;
            }
            read.close()?;
        }
        for buffer in released {
            self.follow_fences(&buffer, stream)?;
        }
        let mut state = lock(&self.state);
        self.reclaim(&mut state)
    }
}
