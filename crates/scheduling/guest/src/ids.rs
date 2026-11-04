//! Opaque identifiers of the objects a policy decides about.
//!
//! The mechanism owns the objects; a policy only compares, remembers and
//! returns these ids. Newtypes over `u64` keep the mechanism's own types out of
//! the interface, so the scheduling crate stays below the engine and a policy
//! module needs no knowledge of it.

use serde::{Deserialize, Serialize};

/// A request the main process is serving.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(pub u64);

/// A KV cache page: one unit of device or host cache storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PageId(pub u64);

/// An encoder cache entry: an encoded input media product.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EncodingId(pub u64);

/// A phase: one block's computation for one kind of call, the unit the
/// scheduler addresses (design §0). The mechanism assigns the ids for an epoch
/// and describes each in [`crate::view::PhaseView`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PhaseId(pub u32);

/// A block of the model: an encoder, denoiser, decoder or causal language
/// model (design §0, §4.3). A block names the computation, not the replica
/// that runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ComponentId(pub u32);
