//! The experts runtime's participation seam: expert-only ranks and model
//! ranks between their own forwards join every expert step of their union
//! (ExpertExchange agreement, join graphs, microbatch rotation).
//!
//! Until the experts runtime is integrated, a rank that must participate
//! fails when the service first asks it to join a step.

use super::super::backend::WorkerBackend;
use crate::Error;
use crate::error::WorkerError;

/// Join one expert step without an own forward. Returns (ran a step,
/// every member left the group).
pub(crate) fn join_step(
    _backend: &WorkerBackend,
    _leaving: bool,
) -> std::result::Result<(bool, bool), WorkerError> {
    Err(WorkerError::classify(
        Error::Unsupported("expert-step participation is served by the experts runtime".into()),
        None,
    ))
}
