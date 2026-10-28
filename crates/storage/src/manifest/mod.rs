//! Copy manifests: how derived state held under one parallel configuration
//! becomes the derived state another configuration needs.
//!
//! Every derived state (a weight, a page of attention state) is described
//! in three layers, the first two declared by the model library:
//!
//! - the logical tensor: its global shape and what each dimension indexes;
//! - its distribution: which index region of it each GPU holds, any number
//!   of GPUs holding the same region ([`region`], [`layout::Logical`]);
//! - its encoding on each GPU: where each held element lies in the bytes
//!   of a storage ([`layout::Field`], [`layout::Encoding`]).
//!
//! A [`Manifest`] is computed once per pair of worlds, the source world
//! that holds the state and the target world that needs it, from both
//! worlds' storage layouts and chunk keys ([`World`]), the GPUs the target
//! may use and the bandwidth of every link between them ([`Request`]):
//!
//! 1. For each target rank and GPU, the bytes the GPU already holds of the
//!    regions the rank needs; the target ranks take the GPUs that maximize
//!    those bytes, and among those the assignment whose busiest link
//!    finishes first.
//! 2. A target chunk whose key a source chunk on the same GPU holds is
//!    mapped read-only ([`Mapping`]): same key, same bytes, granularity
//!    bounds on both sides.
//! 3. Every other region a target chunk holds is divided into blocks that
//!    lie wholly inside or outside each source's region. Each block goes,
//!    largest first, to the source whose link then finishes earliest given
//!    the bytes already assigned to every link (bytes over bandwidth), a
//!    source on the same GPU being a link of its own.
//! 4. Each block becomes byte copies through both sides' encodings
//!    ([`Copy`], strided rows where a dimension repeats with a constant
//!    pitch, merged where adjacent). A field whose encoding is not affine
//!    moves in its canonical encoding: the source converts its group to a
//!    staging space, the target converts its staging space back
//!    ([`Conversion`]); a source of the same encoding and region instead
//!    copies the bytes as they are.
//! 5. Padding the target holds is zeroed ([`Fill`]); regions no source
//!    holds are reported ([`Missing`]) for the target to load.
//!
//! A [`Planner`] caches each pair's executable manifest. Participating
//! workers consume that concrete plan. [`BoundManifest::spans`]
//! turns the copies into the spans of a transfer between the chunks the
//! Storage Server retains ([`crate::protocol::Request::Transfer`]), which
//! its Transfer Server executes. The source world of weights is what the
//! manifests a world retains name ([`weights::sources`]).

pub mod cache;
pub mod layout;
mod plan;
pub mod region;
mod tasks;
#[cfg(test)]
mod tests;
pub mod weights;

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub use layout::{Canonical, Encoding, Field, Logical, Piece, Segment, StorageLayout};
pub use region::Span;
pub use tasks::{BoundManifest, Spans};

use crate::Key;
use crate::protocol::{LinkBandwidth, Memory, Placement};

/// Why a manifest cannot be computed or applied.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A layout or request contradicts itself.
    #[error("invalid copy manifest input: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;

use crate::key::Digest;

/// One chunk of a storage: its bytes and the key its memory is retained
/// under.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Chunk {
    pub offset: u64,
    pub length: u64,
    pub key: Key,
    /// First byte within the retained allocation. Logical cache pages can
    /// borrow several regions of the same retained plane segment.
    pub storage_offset: u64,
    pub binding: Binding,
}

/// Where retained bytes reside, relative to the Transfer Server executing
/// this request. Rank storage follows the rank's assigned GPU; host
/// staging and remote sources carry an explicit physical location.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Binding {
    Rank,
    Local { at: Placement },
    Remote { endpoint: String, at: Placement },
}

impl Binding {
    /// The physical memory owning these bytes. GPU UUIDs have global
    /// identity; host allocations keep their local or remote endpoint.
    pub fn memory(&self, gpu: Option<Gpu>) -> Result<Memory> {
        match self {
            Self::Rank => gpu
                .map(|gpu| Memory::Device { gpu })
                .ok_or_else(|| Error::Invalid("rank storage names no GPU".into())),
            Self::Local {
                at: Placement::Host { .. },
            } => Ok(Memory::Host),
            Self::Local {
                at: Placement::Device { gpu },
            } => Ok(Memory::Device { gpu: *gpu }),
            Self::Remote {
                at: Placement::Device { gpu },
                ..
            } => Ok(Memory::Device { gpu: *gpu }),
            Self::Remote {
                endpoint,
                at: Placement::Host { .. },
            } => Ok(Memory::Remote {
                endpoint: endpoint.clone(),
                gpu: None,
            }),
        }
    }
}

/// One storage of a rank: its layout and its chunks.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Storage {
    pub layout: StorageLayout,
    pub chunks: Vec<Chunk>,
}

impl Storage {
    /// A storage of `layout` in the chunks [`StorageLayout::chunks`] cuts
    /// for `degrees` at `granularity`, keyed with `fingerprint`
    /// ([`StorageLayout::chunk_key`]).
    pub fn chunked(
        layout: StorageLayout,
        granularity: u64,
        degrees: &[u64],
        fingerprint: &dyn Fn(&str) -> Option<[u8; 32]>,
    ) -> Result<Self> {
        layout.validate()?;
        let chunks = layout
            .chunks(granularity, degrees)
            .into_iter()
            .map(|window| {
                Ok(Chunk {
                    key: layout.chunk_key(&window, fingerprint)?,
                    offset: window.start,
                    length: window.end - window.start,
                    storage_offset: 0,
                    binding: Binding::Rank,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layout, chunks })
    }

    /// A storage of `layout` held in pages of `page_bytes` each, as a page
    /// pool's slots hold it: the pages up to the one its last byte lies in,
    /// every page a chunk keyed with `fingerprint`
    /// ([`StorageLayout::chunk_key`]). A page's key thus names the logical
    /// regions its bytes hold and where they lie in it, so a page that two
    /// parallel configurations lay out alike has one key in both, and pages
    /// that hold anything else have others. A layout whose bytes end with
    /// the last byte a field holds leaves no page that holds nothing.
    ///
    /// Fails when the layout declares itself inconsistently or the pages
    /// have no bytes, or when a field's source has no fingerprint.
    pub fn paged(
        layout: StorageLayout,
        page_bytes: u64,
        fingerprint: &dyn Fn(&str) -> Option<[u8; 32]>,
    ) -> Result<Self> {
        layout.validate()?;
        if page_bytes == 0 {
            return Err(Error::Invalid("pages of no bytes hold no storage".into()));
        }
        let pages = layout.bytes.div_ceil(page_bytes);
        let chunks = (0..pages)
            .map(|page| {
                let window = page * page_bytes..(page + 1) * page_bytes;
                Ok(Chunk {
                    key: layout.chunk_key(&window, fingerprint)?,
                    offset: window.start,
                    length: page_bytes,
                    storage_offset: 0,
                    binding: Binding::Rank,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layout, chunks })
    }
}

/// A GPU, by its UUID, as the Storage Server and the Transfer Server name
/// it.
pub type Gpu = [u8; 16];

/// One rank of a world.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rank {
    /// The rank's execution GPU; `None` for a target rank whose GPU is
    /// assigned by the manifest. Each chunk separately binds its memory.
    pub gpu: Option<Gpu>,
    pub storages: Vec<Storage>,
}

/// The ranks of one world under one parallel configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct World {
    pub ranks: Vec<Rank>,
}

impl World {
    /// Whether the same ordered ranks retain the same numerical storage
    /// in the same physical memory. A GPU's local and remote bindings are
    /// views of one allocation, while host endpoints remain distinct.
    pub fn same_storage(&self, other: &Self) -> bool {
        self.ranks.len() == other.ranks.len()
            && self.ranks.iter().zip(&other.ranks).all(|(one, two)| {
                one.gpu == two.gpu
                    && one.storages.len() == two.storages.len()
                    && one.storages.iter().zip(&two.storages).all(|(a, b)| {
                        a.layout == b.layout
                            && a.chunks.len() == b.chunks.len()
                            && a.chunks.iter().zip(&b.chunks).all(|(x, y)| {
                                x.offset == y.offset
                                    && x.length == y.length
                                    && x.key == y.key
                                    && x.storage_offset == y.storage_offset
                                    && x.binding.memory(one.gpu).ok()
                                        == y.binding.memory(two.gpu).ok()
                            })
                    })
            })
    }
}

/// What a manifest is computed from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub source: World,
    pub target: World,
    /// The GPUs the target world's ranks may take, at least one per rank.
    pub gpus: Vec<Gpu>,
    /// The bandwidth of every physical memory link, as the host's Transfer
    /// Server measured it (the listing's `transfer.links`): when it started
    /// for the links within its host, at their first use for the links from
    /// another host's device or host memory (over the NVLink fabric, or READ). A copy within
    /// one GPU takes the link from it to itself, and every link a block may
    /// take must be measured.
    pub links: Vec<LinkBandwidth>,
}

impl Request {
    /// The digest of everything the manifest is computed from.
    pub fn digest(&self) -> Result<[u8; 32]> {
        let encoded = rmp_serde::to_vec(self)
            .map_err(|error| Error::Invalid(format!("encoding a manifest request: {error}")))?;
        let mut digest = Digest::new(b"uniserve-manifest-request");
        digest.bytes(&encoded);
        Ok(digest.finish())
    }
}

/// A target chunk mapped read-only from the source chunk of the same key
/// on the same GPU.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Mapping {
    pub rank: u32,
    pub storage: u32,
    pub chunk: u32,
    pub key: Key,
    pub source_rank: u32,
    pub source_storage: u32,
    pub source_chunk: u32,
}

/// Where a copy reads or writes, on its side's rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Space {
    /// A storage of the rank.
    Storage { storage: u32 },
    /// The staging bytes of [`Manifest::stagings`] entry `staging`.
    Staging { staging: u32 },
}

/// A byte position of one side's rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Endpoint {
    pub rank: u32,
    pub space: Space,
    pub offset: u64,
}

/// `rows` copies of `length` bytes from the source world to the target
/// world, row `r` from `source.offset + r * source_pitch` to
/// `target.offset + r * target_pitch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Copy {
    pub source: Endpoint,
    pub target: Endpoint,
    pub length: u64,
    pub rows: u64,
    pub source_pitch: u64,
    pub target_pitch: u64,
}

impl Copy {
    /// The bytes the copy moves.
    pub fn bytes(&self) -> u64 {
        self.length * self.rows
    }
}

/// Which world a staging space or conversion belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Source,
    Target,
}

/// Bytes a rank holds a field group in, in the group's canonical encoding.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Staging {
    pub side: Side,
    pub rank: u32,
    /// The field group (`Encoding::Encoded::group`).
    pub group: String,
    pub bytes: u64,
}

/// A conversion of a field group between its encoding and its canonical
/// encoding in a staging space: a source converts to canonical before the
/// copies read the staging space, a target converts from canonical after
/// the copies wrote it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Conversion {
    pub staging: u32,
    /// The encoding the group's fields are stored in.
    pub encoding: String,
    /// The storages holding the group's fields, by their tensor names.
    pub fields: Vec<(u32, String)>,
}

/// `rows` runs of `length` zero bytes of a target storage, row `r` at
/// `offset + r * pitch`: padding the target holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Fill {
    pub rank: u32,
    pub storage: u32,
    pub offset: u64,
    pub length: u64,
    pub rows: u64,
    pub pitch: u64,
}

/// A region of a target field no source holds: the target loads it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Missing {
    pub rank: u32,
    pub storage: u32,
    pub tensor: String,
    pub logical: Vec<Span>,
}

/// The bytes a manifest moves over one link and their estimated time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkLoad {
    pub source: Memory,
    pub target: Memory,
    pub bytes: u64,
    pub seconds: f64,
}

/// How a target world obtains the state a source world holds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// The digest of the request the manifest was computed from.
    pub request: [u8; 32],
    /// Each target rank's GPU.
    pub gpus: Vec<Gpu>,
    pub mappings: Vec<Mapping>,
    pub stagings: Vec<Staging>,
    /// Conversions of source groups to canonical, before the copies.
    pub encodes: Vec<Conversion>,
    pub copies: Vec<Copy>,
    /// Conversions of target groups from canonical, after the copies.
    pub decodes: Vec<Conversion>,
    pub fills: Vec<Fill>,
    pub missing: Vec<Missing>,
    pub links: Vec<LinkLoad>,
    /// The digest of everything above.
    pub digest: [u8; 32],
}

impl Manifest {
    /// Compute the manifest of `request`.
    pub fn plan(request: &Request) -> Result<Self> {
        plan::plan(request)
    }

    /// The digest of everything the manifest directs.
    pub fn compute_digest(&self) -> Result<[u8; 32]> {
        let directed = (
            &self.request,
            &self.gpus,
            &self.mappings,
            &self.stagings,
            &self.encodes,
            &self.copies,
            &self.decodes,
            &self.fills,
            &self.missing,
        );
        let encoded = rmp_serde::to_vec(&directed)
            .map_err(|error| Error::Invalid(format!("encoding a manifest: {error}")))?;
        let mut digest = Digest::new(b"uniserve-manifest");
        digest.bytes(&encoded);
        Ok(digest.finish())
    }

    /// The bytes the copies move.
    pub fn copied_bytes(&self) -> u64 {
        self.copies.iter().map(Copy::bytes).sum()
    }

    /// Physical bytes target `rank` must allocate before executing this plan.
    /// Chunks mapped from a retained source need no new allocation.
    pub fn created_bytes(&self, request: &Request, rank: u32) -> u64 {
        let mapped: std::collections::BTreeSet<_> = self
            .mappings
            .iter()
            .filter(|mapping| mapping.rank == rank)
            .map(|mapping| (mapping.storage, mapping.chunk))
            .collect();
        let mut bytes = 0;
        for (storage, held) in request.target.ranks[rank as usize]
            .storages
            .iter()
            .enumerate()
        {
            for (chunk, value) in held.chunks.iter().enumerate() {
                if !mapped.contains(&(storage as u32, chunk as u32)) {
                    bytes += value.length;
                }
            }
        }
        bytes
    }
}

/// Computes each request's manifest once.
#[derive(Default)]
pub struct Planner {
    manifests: HashMap<[u8; 32], Arc<Manifest>>,
}

impl Planner {
    /// The manifest of `request`, computed on its first request.
    pub fn manifest(&mut self, request: &Request) -> Result<Arc<Manifest>> {
        let digest = request.digest()?;
        if let Some(manifest) = self.manifests.get(&digest) {
            return Ok(Arc::clone(manifest));
        }
        let manifest = Arc::new(Manifest::plan(request)?);
        self.manifests.insert(digest, Arc::clone(&manifest));
        Ok(manifest)
    }
}
