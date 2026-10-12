//! The products a video batch's calls read and publish.
//!
//! A call reads products other calls published (`Call::inputs`, a
//! predicate) and publishes its own (`Call::outputs`, a completion scalar).
//! Products live in the rank's [`TensorStore`]: imports from other ranks
//! reach it before the batch executes (`forward::inputs`), and every input
//! is then leased from it ([`Reads`]); host codecs read some inputs in place
//! from their shared-memory exports instead. [`bind_outputs`] reserves a
//! call's outputs at their scheduler placement and [`export`] publishes a
//! written one.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use uniserve_core::CallId;
use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BufferAllocation, BufferId, Call, DType, RequestKey, TensorExport,
    TensorRef, TransferHandle,
};

use super::launch::read_flag;
use crate::ffi::Device;
use crate::storage::OutputPlacement;
use crate::transfer::Transports;
use crate::transfer::exports::Location;
use crate::{Error, EventPool, ProductBuffer, ProductRead, Result, TensorStore};

/// The rank's product owners a batch uses.
#[derive(Clone, Copy)]
pub(crate) struct ProductOwners<'a> {
    pub(crate) store: &'a TensorStore,
    pub(crate) transports: &'a Transports,
    pub(crate) events: &'a Arc<EventPool>,
}

/// One batch's reads of products in the rank's tensor store.
///
/// Products other ranks published were imported and published into the
/// store before the batch executes (`forward::inputs`), so every input is
/// leased like a product of this rank. Inputs a host codec reads in place
/// from their shared-memory exports (*borrowed*) are not in the store.
#[derive(Default)]
pub(crate) struct Reads {
    leases: Vec<Arc<ProductRead>>,
    borrowed: HashSet<BufferId>,
}

impl Reads {
    /// The reads of a batch whose `borrowed` inputs are read in place.
    pub(crate) fn new(borrowed: HashSet<BufferId>) -> Self {
        Self {
            leases: Vec::new(),
            borrowed,
        }
    }

    /// Whether an input is read in place from its export.
    pub(crate) fn borrowed(&self, product: &TensorRef) -> bool {
        self.borrowed.contains(&product.buffer_id())
    }

    /// Lease one input for `consumer`, whole; `stream` waits on its writer.
    pub(crate) fn read(
        &mut self,
        store: &TensorStore,
        product: &TensorRef,
        consumer: CallId,
        device: Device,
        stream: Option<&Stream>,
    ) -> Result<Arc<ProductRead>> {
        let read = store.consume(product, consumer, Some(device), stream)?;
        self.leases.push(Arc::clone(&read));
        if read.region().is_some() {
            return Err(Error::Invalid(
                "media computation requires complete input coverage".into(),
            ));
        }
        Ok(read)
    }

    /// End every lease after the batch's readers were enqueued on `stream`.
    pub(crate) fn complete(&mut self, store: &TensorStore, stream: Option<&Stream>) -> Result<()> {
        let reads = std::mem::take(&mut self.leases);
        if reads.is_empty() {
            return Ok(());
        }
        store.complete_reads(&reads, stream, &[])
    }
}

/// Indices of the calls whose host-gated (U8) predicate is false, read
/// once each on `stream` after its producer.
pub(crate) fn skipped(
    store: &TensorStore,
    plan: &BatchPlan,
    device: Device,
    stream: Option<&Stream>,
) -> Result<HashSet<usize>> {
    let mut skipped = HashSet::new();
    for (index, call) in plan.calls.iter().enumerate() {
        if let Some(predicate) = &call.predicate
            && predicate.dtype == DType::U8
            && !read_flag(store, call, predicate, device.cuda_index(), stream)?
        {
            skipped.insert(index);
        }
    }
    Ok(skipped)
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
    // Issued on `stream` at once, so the export is located here.
    let export = owners.store.export_buffer(
        call,
        product,
        write,
        owners.transports,
        host,
        regions,
        stream,
    )?;
    owners.store.located(write, export)
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
