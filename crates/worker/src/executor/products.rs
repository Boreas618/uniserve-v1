//! The products a media batch's calls read and publish.
//!
//! A call reads products other calls published (`Call::inputs`, a
//! predicate) and publishes its own (`Call::outputs`, a completion scalar).
//! Products live in the rank's [`TensorStore`]: a product published on this
//! rank is leased from it, and one published on other ranks is imported
//! into it through the transports named by the batch's `input_products`
//! before the batch executes. Host codecs instead read some inputs in place
//! from their shared-memory exports (*borrowed* inputs).
//!
//! [`ProductInputs`] holds one batch's imports, leases and the readback of
//! its U8 predicates; [`bind_outputs`] reserves a call's outputs at their
//! scheduler placement and [`publish`] writes, exports and fences one.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::{Arc, Mutex, PoisonError};

use uniserve_core::CallId;
use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BufferAllocation, BufferId, Call, DType, RequestKey, TensorExport,
    TensorRef, TransferHandle,
};

use super::Submission;
use crate::ffi::Device;
use crate::storage::OutputLease;
use crate::storage::OutputPlacement;
use crate::storage::memory::StorageView;
use crate::transfer::exports::Location;
use crate::transfer::{Notify, Transports};
use crate::{Error, EventPool, OutputPool, ProductBuffer, ProductRead, Result, TensorStore};

fn lock<T>(value: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    value.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The rank's product owners a batch uses.
#[derive(Clone, Copy)]
pub(crate) struct ProductOwners<'a> {
    pub(crate) store: &'a TensorStore,
    pub(crate) transports: &'a Transports,
    pub(crate) events: &'a Arc<EventPool>,
}

/// The U8 predicates of a batch, copied to the host before execution.
struct Predicates {
    buffer: OutputLease,
    /// `(call index, host byte)` of each captured predicate.
    rows: Vec<(usize, StorageView)>,
}

/// One batch's product inputs.
#[derive(Default)]
pub(crate) struct ProductInputs {
    /// Imports of products published on other ranks, by buffer.
    imports: HashMap<BufferId, Arc<ProductRead>>,
    /// Imports whose materialization was published to the store.
    completed: HashSet<BufferId>,
    /// Inputs host codecs read in place from their exports.
    borrowed: HashSet<BufferId>,
    /// Leases of local products, completed with the batch's reads.
    leases: Vec<Arc<ProductRead>>,
    predicates: Option<Predicates>,
    submitted: bool,
}

impl ProductInputs {
    pub(crate) fn submitted(&self) -> bool {
        self.submitted
    }

    /// Import every product the batch reads from other ranks, except the
    /// borrowed ones, and copy its U8 predicates to the host.
    ///
    /// `device` is the device a call consumes on and `stream` the stream
    /// its imports and predicate copies are ordered on; `borrowed` selects
    /// inputs a host codec reads in place. Runs once per batch.
    #[expect(
        clippy::too_many_arguments,
        reason = "one batch's complete input sources"
    )]
    pub(crate) fn submit(
        &mut self,
        plan: &BatchPlan,
        owners: ProductOwners<'_>,
        slots: &HashMap<RequestKey, usize>,
        device: Device,
        stream: Option<&Stream>,
        borrowed: impl Fn(&Call, &TensorExport) -> bool,
        outputs: &mut OutputPool,
    ) -> Result<()> {
        if std::mem::replace(&mut self.submitted, true) {
            return Ok(());
        }
        let allocations: HashMap<BufferId, BufferAllocation> = plan
            .buffer_allocations
            .iter()
            .map(|allocation| (allocation.buffer, *allocation))
            .collect();
        let placement = OutputPlacement {
            request_slots: Some(slots),
            allocations: Some(&allocations),
            regions: None,
            shapes: None,
        };
        for export in &plan.input_products {
            let id = export.product.buffer_id();
            let consumers: Vec<&Call> = plan
                .calls
                .iter()
                .filter(|call| {
                    call.tensor_inputs().any(|input| *input == export.product)
                        || call.predicate.as_ref() == Some(&export.product)
                })
                .collect();
            if consumers.is_empty() {
                return Err(Error::Invalid(
                    "a supplied product has no consumer in its batch".into(),
                ));
            }
            if consumers.iter().all(|call| borrowed(call, export)) {
                self.borrowed.insert(id);
                continue;
            }
            if self.imports.contains_key(&id) {
                continue;
            }
            let tensor = export
                .value
                .tensors()
                .first()
                .ok_or_else(|| Error::Invalid("a supplied product has no tensor".into()))?;
            let read = owners.store.import_tensor(
                &export.product,
                tensor,
                device,
                owners.transports,
                placement,
                None,
                stream,
            )?;
            self.imports.insert(id, read);
        }

        let predicated: Vec<(usize, &TensorRef)> = plan
            .calls
            .iter()
            .enumerate()
            .filter_map(|(index, call)| call.predicate.as_ref().map(|value| (index, value)))
            .filter(|(_, value)| value.dtype == DType::U8)
            .collect();
        if !predicated.is_empty() {
            let consumers: Vec<CallId> = plan.calls.iter().map(|call| call.call_id).collect();
            self.capture_predicates(&predicated, &consumers, owners, device, stream, outputs)?;
        }
        Ok(())
    }

    fn capture_predicates(
        &mut self,
        predicated: &[(usize, &TensorRef)],
        consumers: &[CallId],
        owners: ProductOwners<'_>,
        device: Device,
        stream: Option<&Stream>,
        outputs: &mut OutputPool,
    ) -> Result<()> {
        let devices: Vec<i32> = match device {
            Device::Cuda(index) => vec![index],
            Device::Cpu | Device::CudaHost => Vec::new(),
        };
        let buffer = outputs.lease(predicated.len(), predicated.len().max(1), &devices, false)?;
        let mut rows = Vec::with_capacity(predicated.len());
        let mut reads = Vec::new();
        let captured = (|| -> Result<()> {
            let mut value = lock(&buffer);
            for &(index, predicate) in predicated {
                // An imported predicate is copied once its import completed;
                // its value is read like a local lease.
                let read = match self.imports.get(&predicate.buffer_id()) {
                    Some(_) => continue,
                    None => {
                        owners
                            .store
                            .consume(predicate, consumers[index], Some(device), stream)?
                    }
                };
                reads.push(Arc::clone(&read));
                let capture = value.capture_bytes(read.value(), stream, owners.events)?;
                rows.push((index, capture));
            }
            Ok(())
        })();
        let completed = owners.store.complete_reads(&reads, stream, &[]);
        captured?;
        completed?;
        if rows.len() != predicated.len() {
            return Err(Error::Unsupported(
                "a media call's predicate is published on another rank".into(),
            ));
        }
        let fences = {
            let mut value = lock(&buffer);
            let streams: Vec<&Stream> = stream.into_iter().collect();
            value.seal(&streams, owners.events)?
        };
        let fences = fences.unwrap_or_default();
        let finished = Arc::clone(&buffer);
        for fence in &fences {
            owners.events.wake_after(fence)?;
        }
        owners.events.defer(
            fences,
            Box::new(move || {
                lock(&finished).copies_finished();
                Ok(())
            }),
        )?;
        self.predicates = Some(Predicates { buffer, rows });
        Ok(())
    }

    /// Whether every import has arrived and every predicate was copied.
    pub(crate) fn ready(&self) -> Result<bool> {
        for read in self.imports.values() {
            if read.reads().iter().any(|transfer| !transfer.ready()) {
                return Ok(false);
            }
        }
        if let Some(predicates) = &self.predicates
            && lock(&predicates.buffer).ready_at()?.is_none()
        {
            return Ok(false);
        }
        Ok(true)
    }

    /// Notify `submission` when the pending imports arrive. Predicate
    /// copies wake the service through their device fences.
    pub(crate) fn wait(&self, submission: &Arc<Submission>) {
        for read in self.imports.values() {
            for transfer in read.reads() {
                if transfer.ready() {
                    continue;
                }
                let submission = Arc::clone(submission);
                let notify: Notify = Arc::new(move || submission.notify_ready());
                transfer.add_done_callback(notify);
            }
        }
    }

    /// Indices of the calls whose predicate is false; read once the
    /// predicates are ready.
    pub(crate) fn predicated(&self) -> Result<HashSet<usize>> {
        let Some(predicates) = &self.predicates else {
            return Ok(HashSet::new());
        };
        if lock(&predicates.buffer).ready_at()?.is_none() {
            return Err(Error::Invariant(
                "call predicates were read before their copies completed".into(),
            ));
        }
        let mut skipped = HashSet::new();
        for (index, capture) in &predicates.rows {
            // SAFETY: the capture is one pinned host byte whose device copy
            // completed (the buffer's fences are query-ready).
            let value = unsafe { *(capture.address() as *const u8) };
            match value {
                0 => {
                    skipped.insert(*index);
                }
                1 => {}
                _ => {
                    return Err(Error::Invalid(
                        "call predicate is not a canonical boolean".into(),
                    ));
                }
            }
        }
        Ok(skipped)
    }

    /// Whether an input is read in place from its export.
    pub(crate) fn borrowed(&self, product: &TensorRef) -> bool {
        self.borrowed.contains(&product.buffer_id())
    }

    /// Lease one input for `consumer`, whole: an import published on its
    /// first read, or a product of this rank. `stream` waits on its writer.
    pub(crate) fn read(
        &mut self,
        store: &TensorStore,
        product: &TensorRef,
        consumer: CallId,
        device: Device,
        stream: Option<&Stream>,
    ) -> Result<Arc<ProductRead>> {
        let id = product.buffer_id();
        if let Some(read) = self.imports.get(&id) {
            if self.completed.insert(id) {
                store.complete_import(read, stream)?;
            } else {
                store.wait_import(read, stream)?;
            }
            return Ok(Arc::clone(read));
        }
        let read = store.consume(product, consumer, Some(device), stream)?;
        if read.region().is_some() {
            return Err(Error::Invalid(
                "media computation requires complete input coverage".into(),
            ));
        }
        self.leases.push(Arc::clone(&read));
        Ok(read)
    }

    /// End every lease and import after the batch's readers were enqueued
    /// on `stream`.
    pub(crate) fn complete(&mut self, store: &TensorStore, stream: Option<&Stream>) -> Result<()> {
        let mut reads: Vec<Arc<ProductRead>> = std::mem::take(&mut self.leases);
        reads.extend(self.imports.drain().map(|(_, read)| read));
        self.completed.clear();
        if reads.is_empty() {
            return Ok(());
        }
        store.complete_reads(&reads, stream, &[])
    }
}

/// The scheduler placement of a call's outputs on this rank: buffer-pool
/// allocations, request relay slots, and per output the logical shape and
/// this rank's region of it.
pub(crate) struct Placement<'a> {
    pub(crate) allocations: &'a HashMap<BufferId, BufferAllocation>,
    pub(crate) slots: &'a HashMap<RequestKey, usize>,
}

/// One output this rank publishes: its logical shape and, for a sharded
/// product, the region this rank holds.
pub(crate) struct OutputShape {
    pub(crate) product: TensorRef,
    pub(crate) shape: Vec<u64>,
    pub(crate) region: Option<Vec<Range<u64>>>,
}

/// Reserve a call's persistent outputs at their placement on `device`.
pub(crate) fn bind_outputs(
    store: &TensorStore,
    outputs: &[OutputShape],
    device: Device,
    placement: &Placement<'_>,
) -> Result<Vec<ProductBuffer>> {
    if outputs.is_empty() {
        return Ok(Vec::new());
    }
    let shapes: HashMap<BufferId, Vec<u64>> = outputs
        .iter()
        .map(|output| (output.product.buffer_id(), output.shape.clone()))
        .collect();
    let regions: HashMap<BufferId, Vec<Range<u64>>> = outputs
        .iter()
        .filter_map(|output| {
            output
                .region
                .clone()
                .map(|region| (output.product.buffer_id(), region))
        })
        .collect();
    let bindings: Vec<(TensorRef, Device)> = outputs
        .iter()
        .map(|output| (output.product.clone(), device))
        .collect();
    store.bind_outputs(
        &bindings,
        OutputPlacement {
            request_slots: Some(placement.slots),
            allocations: Some(placement.allocations),
            regions: Some(&regions),
            shapes: Some(&shapes),
        },
    )
}

/// Reserve a call's completion scalar in its request's relay lane.
pub(crate) fn bind_completion(
    store: &TensorStore,
    call: &Call,
    device: Device,
    slots: &HashMap<RequestKey, usize>,
) -> Result<Option<ProductBuffer>> {
    let Some(product) = &call.completion_output else {
        return Ok(None);
    };
    let bound = store.bind_outputs(
        &[(product.clone(), device)],
        OutputPlacement {
            request_slots: Some(slots),
            ..OutputPlacement::default()
        },
    )?;
    Ok(bound.into_iter().next())
}

/// Export a written product to the call's consumers. `regions` limits the
/// export to parts of the value (encoded rows publish their prefixes only).
pub(crate) fn export(
    owners: ProductOwners<'_>,
    call: &Call,
    product: &TensorRef,
    write: &ProductBuffer,
    host: bool,
    regions: Option<&[Vec<Range<u64>>]>,
    stream: Option<&Stream>,
) -> Result<(TensorExport, Vec<Location>)> {
    owners.store.export_buffer(
        call,
        product,
        write,
        owners.transports,
        host,
        regions,
        stream,
    )
}

/// The tensor descriptor of a supplied input product.
pub(crate) fn supplied<'a>(plan: &'a BatchPlan, product: &TensorRef) -> Option<&'a TensorExport> {
    plan.input_products
        .iter()
        .find(|export| export.product == *product)
}

/// The published tensor of a device or encoder product export.
pub(crate) fn tensor(export: &TensorExport) -> Result<&uniserve_ipc::TensorTransfer> {
    match &export.value {
        TransferHandle::DeviceProduct { tensor, .. } | TransferHandle::Encoder { tensor, .. } => {
            Ok(tensor)
        }
        TransferHandle::Latent { .. } => Err(Error::Invalid(
            "a media input is not a latent trajectory".into(),
        )),
    }
}
