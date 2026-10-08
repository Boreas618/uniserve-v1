//! The host media runtime's execution seam: media reading, video and audio encoding and muxing on the host lane (MuxSession, codecs).
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
    Err(super::unsupported(kind, "host media"))
}
