//! Publication of a batch's products to the ranks and workers that read
//! them.
//!
//! A runtime exports a product once the producer has written it: the
//! tensor store exports its value through every export mechanism that
//! reaches the producing call's consumers, retains each location's
//! retirement on the product's buffer, and the product's descriptor joins
//! the call's completion (and so the batch result the engine routes to the
//! consumers). The locations become the buffer's registered exports at the
//! batch's commit point and are revoked when the buffer is released; a
//! batch that fails before committing revokes them at once. The same rules
//! hold for every runtime, so this module is the one place that applies
//! them.
//!
//! The products a video batch's calls read and publish:
//!
//! A call reads products other calls published (`Call::inputs`, a
//! predicate) and publishes its own (`Call::outputs`, a completion scalar).
//! Products live in the rank's [`ProductStore`]: imports from other ranks
//! reach it before the batch executes ([`crate::executor::inputs`]), and
//! every input is then leased from it
//! ([`crate::executor::video::MediaBatch::read`]); host codecs read some
//! inputs in place from their shared-memory exports instead.
//! [`bind_outputs`] reserves a call's outputs at their scheduler placement
//! and [`export`] publishes a written one.

use crate::executor::PendingOutput;
use crate::ffi::Device;
use crate::storage::{OutputPlacement, ProductBuffer, ProductExport, ProductStore};
use crate::stream::Issue;
use crate::transfer::Transports;
use crate::transfer::exports::{self, Location};
use crate::{Error, Result};
use std::collections::HashMap;
use std::ops::Range;
use uniserve_cuda::Stream;
use uniserve_ipc::{
    Batch as BatchPlan, BufferAllocation, BufferId, Call, RequestKey, TensorExport, TensorRef,
    TransferHandle,
};

/// The product exports a batch accepted before its commit point.
#[derive(Default)]
pub(crate) struct Publications {
    exports: Vec<(BufferId, Vec<Location>)>,
}

impl Publications {
    /// Accept an export another owner made of `buffer` (a latent version,
    /// say): its `locations` register with the batch's products and its
    /// descriptor joins the call's completion products.
    pub(crate) fn accept(
        &mut self,
        buffer: BufferId,
        locations: Vec<Location>,
        export: TensorExport,
        output: &mut PendingOutput,
    ) -> Result<()> {
        // Record the locations before the descriptor is accepted, so a
        // failure from here on revokes them with the batch.
        self.exports.push((buffer, locations));
        let mut products = output.products().to_vec();
        products.push(export);
        output.set_products(products)
    }

    /// Issue the export of an encoder feature to other ranks and workers to
    /// `stream`, behind the work that produced it. Readers on this rank
    /// lease it from the tensor store directly, so a feature is exported
    /// only when the rank publishes over a mechanism other ranks read, and
    /// only by the rank that reports its component's results: `None`
    /// otherwise. [`Self::locate`] publishes the export once that work was
    /// performed.
    pub(crate) fn issue_feature(
        tensors: &ProductStore,
        transports: &Transports,
        call: &Call,
        write: &ProductBuffer,
        reports: bool,
        stream: impl Issue,
    ) -> Result<Option<ProductExport>> {
        if !reports || !transports.exports_remotely() {
            return Ok(None);
        }
        let product = call
            .encoder_output
            .as_ref()
            .filter(|product| product.buffer_id() == write.id())
            .ok_or_else(|| {
                Error::Invariant("an exported feature names no encoder output".into())
            })?;
        tensors
            .export_buffer(call, product, write, transports, false, None, stream)
            .map(Some)
    }

    /// Publish an issued `export` of `write` once its work was performed:
    /// its locations register with the batch's products and its descriptor
    /// joins the call's completion products.
    pub(crate) fn locate(
        &mut self,
        tensors: &ProductStore,
        write: &ProductBuffer,
        export: ProductExport,
        output: &mut PendingOutput,
    ) -> Result<()> {
        let (export, locations) = tensors.located(write, export)?;
        self.accept(write.id(), locations, export, output)
    }

    /// Check, before the commit point, that no buffer already registers
    /// different locations.
    pub(crate) fn validate(&self, tensors: &ProductStore) -> Result<()> {
        for (buffer, locations) in &self.exports {
            tensors.validate_exports(*buffer, locations)?;
        }
        Ok(())
    }

    /// Register the locations with their buffers at the commit point; they
    /// are revoked when the buffers are released.
    pub(crate) fn register(&mut self, tensors: &ProductStore) -> Result<()> {
        for (buffer, locations) in self.exports.drain(..) {
            tensors.register_exports(buffer, locations)?;
        }
        Ok(())
    }

    /// Revoke every location of a batch that failed before its commit
    /// point; physical accesses still retire through the transports.
    pub(crate) fn revoke(&mut self) -> Result<()> {
        let locations: Vec<Location> = self
            .exports
            .drain(..)
            .flat_map(|(_, locations)| locations)
            .collect();
        exports::release(&locations)
    }
}

/// The rank's product owners a batch publishes through.
#[derive(Clone, Copy)]
pub(crate) struct ProductOwners<'a> {
    pub(crate) store: &'a ProductStore,
    pub(crate) transports: &'a Transports,
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
    store: &ProductStore,
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
