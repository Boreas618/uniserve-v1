//! The words of the serving options that the engine itself interprets: a closed vocabulary a
//! serving option names by a word (a standby's storage, an element type).
//!
//! The daemon parses its options into these values, so a word no engine takes is refused while
//! the options are read, naming its option. A word that selects a package (a scheduling policy,
//! an expert exchange, an attention kernel) is not among them: it is the package's backend name,
//! which the Build Server resolves among the packages it holds and refuses when it holds none.

use serde::{Deserialize, Serialize};

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
