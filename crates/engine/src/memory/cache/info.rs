//! What a rank reports of its KV cache planes: their layout, the segments they are made of, and
//! where each unit's bytes lie for copies between processes.

use proto::request::PrefixCacheDtype;
use serde::{Deserialize, Serialize};

use crate::batch::cache::CacheGroup;

/// Physical cache unit pool exposed by a worker that executes AR work.
///
/// The pool is `num_units` allocation units, unit zero being the padding
/// sentinel. Each group's logical pages draw `units_per_page` units from the
/// one pool, so the scheduler accounts every group's demand in units. Each
/// group also places this rank's share of the model's logical cache: its
/// layers by global id and its KV heads by offset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrefixCacheInfo {
    /// Allocation units in the pool, including the unit-zero sentinel.
    pub num_units: u32,
    /// Element data type of cache tensors.
    pub dtype: PrefixCacheDtype,
    /// Cache groups in table order.
    pub groups: Vec<CacheGroup>,
    /// Where the bytes of every unit lie in the rank's retained device
    /// storage, run by run, for copies between processes (a prefill/decode
    /// transfer); empty for planes not retained at a Storage Server.
    #[serde(default)]
    pub plane_runs: Vec<CachePlaneRun>,
    /// Units `1..=scratch_units` the rank's warmup lends to its startup
    /// forwards. A standby worker warms up on them while its predecessor
    /// serves, so the engine never allocates them.
    pub scratch_units: u32,
    /// The segments the rank's value and state planes are made of, in unit order, each
    /// of whole units that give storage back and are created again whole;
    /// empty for planes not split.
    #[serde(default)]
    pub segments: Vec<CacheSegment>,
    /// The segments the planes hold, `[0, held_segments)`: the engine
    /// allocates no unit of the segments after, whose storage was given back.
    #[serde(default)]
    pub held_segments: u32,
    /// The units the planes map read-only from the predecessor epoch's
    /// planes, `[inherited_start, inherited_start + inherited_units)`: they
    /// hold the predecessor's pages, which the engine reads once its index
    /// is imported and allocates only once the predecessor is gone and the
    /// rank took its storage; zero for planes of their own.
    #[serde(default)]
    pub inherited_start: u32,
    #[serde(default)]
    pub inherited_units: u32,
    /// The page identity of each cache group, in group order: `|`-separated
    /// stages naming the weights that compute the group's pages and the
    /// pages' layout. Pipeline stages union these stages; an unrelated
    /// group or a downstream output head changes none of a group's keys. The
    /// engine seeds the group's page keys from it at every cache level
    /// (`storage::key::page_seed`); empty for a consumer without a model.
    #[serde(default)]
    pub identities: Vec<String>,
    /// Where an engine on another host reaches the rank's host's Storage
    /// Server (`StorageLaunch::tcp`); `None` on the engine's host.
    #[serde(default)]
    pub storage_endpoint: Option<String>,
}

/// One segment of a rank's KV cache planes: an allocation of whole units on
/// every plane, which an epoch created and the Storage Server retains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheSegment {
    /// The epoch that created the segment's storage.
    pub epoch: u64,
    /// The segment's ordinal among those its epoch created.
    pub ordinal: u32,
    /// The units the segment holds.
    pub units: u32,
}

/// The exact segments a rank's KV cache planes are made of, and how many of
/// the leading ones exist: a relaunched rank of the same epoch, or a
/// standby rank group's, lays its planes out alike and takes them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CacheTable {
    pub segments: Vec<CacheSegment>,
    /// The segments `[0, held)` that exist; the others were given back.
    pub held: u32,
}

/// A cache group's converted loads on one rank: the pages of `source`, a
/// dense representation of the same logical pages, are read from a peer
/// host into slots of the rank's staging tensor and converted on the device
/// into the group's own representation as they are loaded. They are never
/// committed to a host pool in the source representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PoolConversion {
    /// The shape of the source pages: the layout of the dense
    /// representation, the rank's kv head slice and the bytes of one page.
    pub source: storage::protocol::PageShape,
    /// The rank's staging tensor, an exclusive host tensor retained at its
    /// host's Storage Server, into which a peer read writes source pages.
    pub staging: storage::Key,
    /// The source pages the staging tensor holds; slot `s` lies at
    /// `s * source.page_bytes`.
    pub slots: u32,
}

/// One run of a rank's device cache planes, as a copy between processes
/// addresses it: for each unit `u` in `first..first + count`, `bytes` bytes
/// at `offset + (u - first) * bytes` of the retained exclusive tensor `key`
/// on the GPU `gpu`.
///
/// The runs of one `field` (`prefix_cache.key.values/0` for the key values
/// of a page's column 0, and so on for the values, the initialization flags,
/// the FP8 scales and the state checkpoints) hold that field's bytes of every
/// unit once. Two ranks of the same cache layout name the same fields at the
/// same bytes per unit, so a unit copies between them field by field, from
/// the run of one holding the unit to the run of the other holding it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePlaneRun {
    pub field: String,
    pub key: storage::Key,
    pub gpu: [u8; 16],
    pub first: u32,
    pub count: u32,
    pub offset: u64,
    pub bytes: u64,
}
