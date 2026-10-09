//! Execution-lane settings shared by the launcher and rank runtime.

use std::collections::HashSet;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{CallKind, ForwardMode, MediaCall, TransferMode};

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
    #[error("invalid execution lane JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unknown execution lane domain {0:?}")]
    Domain(String),
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

impl FromStr for LaneConfig {
    type Err = LaneConfigError;

    /// Resolve the public `--lane` domain selectors to concrete operations.
    /// Rank launch descriptors carry the resolved structure directly.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Options {
            lane_id: String,
            sm_budget: usize,
            domains: Vec<String>,
            max_batch_calls: Option<usize>,
            max_batch_tokens: Option<usize>,
            max_inflight: Option<usize>,
        }
        let options: Options = serde_json::from_str(value)?;
        let mut call_kinds = Vec::new();
        for domain in options.domains {
            use CallKind::{Forward, Media, Transfer};
            match domain.as_str() {
                "prefill" => {
                    call_kinds.extend([
                        Forward(ForwardMode::Prefill),
                        Media(MediaCall::VisionEncoding),
                        Media(MediaCall::LatentEncoding),
                        Media(MediaCall::TextEncoding),
                    ]);
                    call_kinds.extend(TransferMode::ALL.map(Transfer));
                }
                "decode" => call_kinds.extend([
                    Forward(ForwardMode::Decode),
                    Forward(ForwardMode::Verify),
                    Forward(ForwardMode::TokenDenoising),
                ]),
                "flow" => call_kinds.extend(
                    [
                        MediaCall::LatentPreparation,
                        MediaCall::Denoising,
                        MediaCall::ImageDecoding,
                        MediaCall::VideoDecoding,
                        MediaCall::AudioDecoding,
                        MediaCall::VideoEncoding,
                        MediaCall::AudioEncoding,
                        MediaCall::Muxing,
                    ]
                    .map(Media),
                ),
                _ => return Err(LaneConfigError::Domain(domain)),
            }
        }
        let lane = Self {
            lane_id: options.lane_id,
            sm_budget: options.sm_budget,
            call_kinds,
            max_batch_calls: options.max_batch_calls,
            max_batch_tokens: options.max_batch_tokens,
            max_inflight: options.max_inflight,
        };
        lane.validate()?;
        Ok(lane)
    }
}
