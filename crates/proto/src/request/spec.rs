//! The words of the serving options: the closed vocabularies a serving option
//! names by a word (a policy, a backend, an element type).
//!
//! The daemon parses its options into these values, so a word no engine
//! takes is refused while the options are read, naming its option. The engine
//! configuration (`config::EngineConfig`) carries each value to the engine
//! processes, and their interface with their workers (`ipc`) carries it on to
//! the ranks. A new word is an addition: a daemon that does not know it never
//! names it.

use serde::{Deserialize, Serialize};

/// Policy used to order requests waiting for admission.
///
/// It orders the token-request queue only; media requests wait in arrival
/// order under either policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulingPolicy {
    /// Orders requests by arrival time.
    #[default]
    Fcfs,
    /// Orders requests by ascending priority value, then arrival time.
    Priority,
}

/// How replicas access distributed experts: token exchange or weight prefetch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExpertExchange {
    /// FlashInfer's NVLink all-to-all dispatch and combine around each
    /// rank's grouped expert kernel.
    #[default]
    #[serde(rename = "alltoall")]
    AllToAll,
    /// The fused CuTeDSL MegaMoE kernel, which dispatches, runs the experts
    /// and combines in one launch per layer over NVSHMEM; NVFP4 experts only.
    #[serde(rename = "megamoe")]
    MegaMoe,
    /// Elastic source/expert dispatch and combine with grouped expert compute.
    #[serde(rename = "deepep")]
    DeepEp,
    /// Independent data-parallel execution with asynchronous peer weight
    /// prefetch into double buffers. Inference has no rank collectives.
    #[serde(rename = "dwdp")]
    Dwdp,
}

/// FlashInfer implementation a rank's attention runs on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashInferBackend {
    /// Lets the worker select a compatible backend.
    #[default]
    Auto,
    /// Uses FlashAttention 2 kernels.
    Fa2,
    /// Uses FlashAttention 3 kernels.
    Fa3,
}

impl FlashInferBackend {}

impl std::str::FromStr for FlashInferBackend {
    type Err = FlashInferBackendParseError;

    /// Parses the spelling [`FlashInferBackend::as_str`] produces.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "fa2" => Ok(Self::Fa2),
            "fa3" => Ok(Self::Fa3),
            _ => Err(FlashInferBackendParseError(value.to_owned())),
        }
    }
}

/// Error returned for an unsupported FlashInfer backend name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported FlashInfer backend {0:?}")]
pub struct FlashInferBackendParseError(String);

/// How a parked standby worker keeps the device storage its graphs and
/// scratch address.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StandbyStorage {
    /// Resident while every device keeps beside it the share of its storage
    /// a serving worker leaves for graphs and activations, suspended
    /// otherwise.
    #[default]
    Auto,
    /// Stays mapped while the standby parks.
    Resident,
    /// Moves to host storage while the standby parks, keeping its addresses,
    /// and back at the promotion.
    Suspended,
}

crate::tensor::dtype_subset! {
    /// The element type of the paged key/value cache (`kv_cache_config.dtype`): a subset of the
    /// interfaces' element types ([`crate::tensor::DType`]), named by its names.
    PrefixCacheDtype, "KV-cache dtype" {
        /// IEEE 754 half precision.
        Float16 = F16,
        /// Brain floating-point half precision.
        BFloat16 = BF16,
        /// IEEE 754 single precision.
        Float32 = F32,
        /// Finite-only E4M3 8-bit floating point.
        Float8E4m3Fn = F8E4M3,
    }
}
