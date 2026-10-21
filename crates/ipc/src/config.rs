//! Execution-lane settings shared by the engine and the worker runtime.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::CallKind;

/// Why an execution lane's settings are refused.
#[derive(Debug, thiserror::Error)]
pub enum LaneConfigError {
    #[error("lane id must be a non-empty token")]
    LaneId,
    #[error("lane SM budget must be positive")]
    SmBudget,
    #[error("lane call kinds must be non-empty and unique")]
    CallKinds,
    /// The named optional limit is configured as zero.
    #[error("lane {0} must be positive when configured")]
    Limit(&'static str),
}

/// Concrete operations and resource limits assigned to one CUDA lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaneConfig {
    pub lane_id: String,
    /// Number of streaming multiprocessors in the lane's partition.
    pub sm_budget: usize,
    pub call_kinds: Vec<CallKind>,
    pub max_batch_calls: Option<usize>,
    pub max_batch_tokens: Option<usize>,
    pub max_inflight: Option<usize>,
}

impl LaneConfig {
    /// Reject ambiguous operations and limits that cannot allocate a lane.
    pub fn validate(&self) -> Result<(), LaneConfigError> {
        if self.lane_id.is_empty() || self.lane_id.chars().any(char::is_whitespace) {
            return Err(LaneConfigError::LaneId);
        }
        if self.sm_budget == 0 {
            return Err(LaneConfigError::SmBudget);
        }
        if self.call_kinds.is_empty()
            || self.call_kinds.iter().collect::<HashSet<_>>().len() != self.call_kinds.len()
        {
            return Err(LaneConfigError::CallKinds);
        }
        for (name, value) in [
            ("max_batch_calls", self.max_batch_calls),
            ("max_batch_tokens", self.max_batch_tokens),
            ("max_inflight", self.max_inflight),
        ] {
            if value == Some(0) {
                return Err(LaneConfigError::Limit(name));
            }
        }
        Ok(())
    }
}
