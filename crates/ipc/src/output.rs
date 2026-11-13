//! What a call's completion states besides its tokens: its status and the
//! latent page layouts it reports. The completion itself is
//! [`crate::execution::CallOutput`]; its error class, timing and media
//! artifact are the scheduler's batch's ([`crate::batch`]).

use super::*;

/// Terminal status of one call's completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum CallStatus {
    /// Execution completed successfully.
    Ok = 0,
    /// The call was skipped because its predicate was false.
    Predicated = 1,
    /// Execution failed with a deterministic error code.
    Error = 2,
}

/// How one denoising rank's request slot pages hold a video request's
/// latent samples.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankLatentLayout {
    /// The rank's index in its worker group.
    pub rank: u32,
    /// The slot's pages as one storage (`storage::manifest::StorageLayout`,
    /// JSON): the logical sample tensors, the part of each the rank holds
    /// and where its values lie in the pages.
    pub layout: String,
}
