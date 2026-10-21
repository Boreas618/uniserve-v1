//! The words of an engine spec: the closed vocabularies a serving option
//! names by a word (a policy, a backend, an element type).
//!
//! The daemon parses its options into these values, so a word no engine
//! takes is refused while the options are read, naming its option. The spec
//! carries each as its word ([`EngineSpec`](super::wire::EngineSpec)), which
//! the engine parses back into the same value and its interface with its
//! workers (`ipc`) carries on to the ranks. A new word is an addition: a
//! daemon that does not know it never names it.

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

impl FlashInferBackend {
    /// Returns the stable command-line spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fa2 => "fa2",
            Self::Fa3 => "fa3",
        }
    }
}

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

/// Which GPU computations a rank captures as CUDA graphs at startup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphPolicy {
    /// Every call serves eagerly.
    Off,
    /// Captures graphs unless serving does not capture the attention
    /// provider's calls, and then serves eagerly.
    #[default]
    Auto,
    /// Requires graphs and refuses an attention provider serving does not
    /// capture.
    Full,
}

/// How a rank reads its checkpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadFormat {
    /// Detects the checkpoint files' own format.
    #[default]
    Auto,
    /// Reads safetensors files.
    Safetensors,
    /// Reads PyTorch pickle files.
    Pt,
    /// Initializes weights without reading the checkpoint.
    Dummy,
    /// Reads the checkpoint one layer at a time.
    Layered,
}

/// Storage dtype used by paged key/value cache tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefixCacheDtype {
    /// IEEE 754 half precision.
    #[serde(rename = "float16")]
    Float16,
    /// Brain floating-point half precision.
    #[serde(rename = "bfloat16")]
    BFloat16,
    /// IEEE 754 single precision.
    #[serde(rename = "float32")]
    Float32,
    /// Finite-only E4M3 8-bit floating point.
    #[serde(rename = "float8_e4m3fn")]
    Float8E4m3Fn,
}

impl PrefixCacheDtype {
    /// Returns the stable wire name for this dtype.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
            Self::Float8E4m3Fn => "float8_e4m3fn",
        }
    }
}

impl std::str::FromStr for PrefixCacheDtype {
    type Err = PrefixCacheDtypeParseError;

    /// Parses a stable prefix-cache dtype wire name.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "float16" => Ok(Self::Float16),
            "bfloat16" => Ok(Self::BFloat16),
            "float32" => Ok(Self::Float32),
            "float8_e4m3fn" => Ok(Self::Float8E4m3Fn),
            _ => Err(PrefixCacheDtypeParseError(value.to_owned())),
        }
    }
}

impl std::fmt::Display for PrefixCacheDtype {
    /// Writes the stable prefix-cache dtype wire name.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Error returned for an unsupported prefix-cache dtype string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported KV-cache dtype {0:?}")]
pub struct PrefixCacheDtypeParseError(String);
