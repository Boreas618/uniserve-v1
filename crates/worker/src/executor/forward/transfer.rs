//! The transfer runtime's execution seam: tensor, KV export and KV install transfers (TransferPool, KVImporter).
//!
//! The runtime owns its runners and bindings (registered in
//! [`crate::model::registry::Components`]), its input preparation and its
//! result materialization. A KV export to the image generation
//! destination (an image's conditioning) is served by the image runtime;
//! other transfer kinds fail the batch as unsupported until the transfer
//! runtime is integrated on this rank.

use uniserve_ipc::{CallKind, TransferMode};

use super::super::backend::{WorkerBackend, WorkerBatch};
use crate::error::WorkerError;

/// Execute a batch of `kind` calls.
pub(crate) fn execute(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<(), WorkerError> {
    if kind == CallKind::Transfer(TransferMode::KvExport) && backend.image_mut().is_some() {
        return super::image::execute(backend, batch, kind);
    }
    Err(super::unsupported(kind, "transfer"))
}
