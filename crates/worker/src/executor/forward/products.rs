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
use uniserve_ipc::{BufferId, Call, TensorRef};

use crate::executor::PendingOutput;
use crate::storage::{ProductBuffer, TensorStore};
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
        let (export, locations) =
            tensors.export_buffer(call, product, write, transports, false, None, stream)?;
        // Record the locations before the descriptor is accepted, so a
        // failure from here on revokes them with the batch.
        self.exports.push((write.id(), locations));
        let mut products = output.products().to_vec();
        products.push(export);
        output.set_products(products)
    }

    /// Export an encoder feature to other ranks and workers. Readers on this
    /// rank lease it from the tensor store directly, so a feature is
    /// exported only when the rank publishes over a mechanism other ranks
    /// read, and only by the rank that reports its component's results.
    #[expect(
        clippy::too_many_arguments,
        reason = "one feature export's complete inputs"
    )]
    pub(crate) fn export_feature(
        &mut self,
        tensors: &TensorStore,
        transports: &Transports,
        call: &Call,
        write: &ProductBuffer,
        reports: bool,
        stream: Option<&Stream>,
        output: &mut PendingOutput,
    ) -> Result<()> {
        if !reports || !transports.exports_remotely() {
            return Ok(());
        }
        let product = call
            .encoder_output
            .as_ref()
            .filter(|product| product.buffer_id() == write.id())
            .ok_or_else(|| {
                Error::Invariant("an exported feature names no encoder output".into())
            })?;
        self.export(tensors, transports, call, product, write, stream, output)
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
