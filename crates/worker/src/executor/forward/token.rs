//! The text runtime's execution seam: prefill, decode and speculative-verify batches of causal language models (TextRunner: KV-cache attention, sampling, graph replay).
//!
//! The runtime owns its runners and bindings (registered in
//! [`crate::model::registry::Components`]), its input preparation and its
//! result materialization; until it is integrated on this rank, its call
//! kinds fail the batch as unsupported.

use uniserve_ipc::CallKind;

use super::super::backend::{WorkerBackend, WorkerBatch};
use crate::error::WorkerError;

/// Execute a batch of `kind` calls.
pub(crate) fn execute(
    _backend: &mut WorkerBackend,
    _batch: &mut WorkerBatch,
    kind: CallKind,
) -> std::result::Result<(), WorkerError> {
    Err(super::unsupported(kind, "text"))
}
