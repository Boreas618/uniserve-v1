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

use uniserve_cuda::Stream;
use uniserve_ipc::{BufferId, Call, TensorExport, TensorRef};

use crate::executor::PendingOutput;
use crate::storage::{ProductBuffer, ProductExport, TensorStore};
use crate::stream::Issue;
use crate::transfer::Transports;
use crate::transfer::exports::{self, Location};
use crate::{Error, Result};

/// The product exports a batch accepted before its commit point.
#[derive(Default)]
pub(crate) struct Publications {
    exports: Vec<(BufferId, Vec<Location>)>,
}

impl Publications {
    /// Export `write`, the produced value of `call`'s output `product`, on
    /// `stream` (which produced it), and add its descriptor to the call's
    /// completion products.
    #[expect(
        clippy::too_many_arguments,
        reason = "one product export's complete inputs"
    )]
    pub(crate) fn export(
        &mut self,
        tensors: &TensorStore,
        transports: &Transports,
        call: &Call,
        product: &TensorRef,
        write: &ProductBuffer,
        stream: Option<&Stream>,
        output: &mut PendingOutput,
    ) -> Result<()> {
        // Issued on `stream` at once, so the export is located here.
        let export =
            tensors.export_buffer(call, product, write, transports, false, None, stream)?;
        let (export, locations) = tensors.located(write, export)?;
        self.accept(write.id(), locations, export, output)
    }

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
        tensors: &TensorStore,
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
        tensors: &TensorStore,
        write: &ProductBuffer,
        export: ProductExport,
        output: &mut PendingOutput,
    ) -> Result<()> {
        let (export, locations) = tensors.located(write, export)?;
        self.accept(write.id(), locations, export, output)
    }

    /// Check, before the commit point, that no buffer already registers
    /// different locations.
    pub(crate) fn validate(&self, tensors: &TensorStore) -> Result<()> {
        for (buffer, locations) in &self.exports {
            tensors.validate_exports(*buffer, locations)?;
        }
        Ok(())
    }

    /// Register the locations with their buffers at the commit point; they
    /// are revoked when the buffers are released.
    pub(crate) fn register(&mut self, tensors: &TensorStore) -> Result<()> {
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
