//! The weight manifests ranks retain, and the source epoch they make.
//!
//! A rank that retains its weights at its host's Storage Server retains,
//! beside its chunks, a manifest: host bytes of JSON text under the name
//! [`NAME`] that state the manifest's [`VERSION`], the digest of what
//! determines each tensor's bytes (`fingerprints`, by tensor name),
//! the rank's tensors as its library describes them, and
//! its storages ([`RetainedStorage`]): each one's device among the rank's
//! devices, layout and chunks. The worker writes it; the ranks of an epoch
//! that receives weights and the ranks a daemon plans its preparation with
//! read the storages of every manifest an epoch retains as the source epoch
//! of a copy manifest ([`sources`]), all from the same listing and texts and
//! by the fingerprints of the whole model, so all compute the same source
//! epoch.

use serde::{Deserialize, Serialize};

use super::{Epoch, Storage};
use crate::protocol::LinkBandwidth;

/// The name a rank's manifest is retained under, by which the manifests an
/// epoch retains are found.
pub const NAME: &str = "weights manifest";

/// The version of the manifest the worker writes and reads; a manifest of
/// another version names no storage.
pub const VERSION: u64 = 7;

/// One retained storage of a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetainedStorage {
    /// The index of the storage's device among the rank's devices; `None`
    /// for the host region of a stack streamed from host storage.
    pub device: Option<usize>,
    #[serde(flatten)]
    pub storage: Storage,
}

/// What a successor rank must hold on its GPU before it can serve.
/// Weight bytes are filled from the concrete transfer manifest; the other
/// quantities come from the target rank's numerical and worker layouts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankBudget {
    pub quota: u64,
    pub process: u64,
    pub graphs: u64,
    pub fixed: u64,
    pub cache: u64,
    pub cache_capacity: u64,
    /// Bytes of one target cache tail segment, its physical allocation unit.
    pub cache_segment: u64,
    pub weights: u64,
    pub pages: Vec<CachePage>,
    /// A complete cross-epoch request reservation, when the caller routes
    /// requests from multiple predecessor engines onto this target GPU.
    pub in_flight: Option<CacheReservation>,
}

/// Value bytes retained for accepted prefixes and the device segments
/// required to materialize them. Host copies need only the value bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheReservation {
    pub bytes: u64,
    pub mapped: u64,
}

/// Value storage added by a target cache group's logical page, including its
/// resident layers, heads and numerical encoding. Full-pool metadata and unit
/// index columns are already charged to the startup cache budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePage {
    pub tokens: u32,
    pub bytes: u64,
    pub window: Option<u32>,
    pub state: bool,
}

/// The weights one epoch retains on a host, as the source epoch of a copy
/// manifest, and the links its Transfer Server measured between the GPUs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sources {
    pub epoch: Epoch,
    pub links: Vec<LinkBandwidth>,
}

/// A daemon's executable weight redistribution plan. Main processes consume this
/// exact request and manifest; they do not independently choose another plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transfer {
    pub request: super::Request,
    pub manifest: super::Manifest,
}
