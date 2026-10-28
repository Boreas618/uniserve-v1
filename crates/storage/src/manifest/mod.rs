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
//! The manifest's digest covers everything it directs; every participating
//! worker verifies it before executing ([`Manifest::verify`]), and a
//! [`Planner`] computes each pair's manifest once. [`Manifest::spans`]
//! turns the copies into the spans of a transfer between the chunks the
//! Storage Server retains ([`crate::protocol::Request::Transfer`]), which
//! its Transfer Server executes.

pub mod layout;
mod plan;
pub mod region;
mod tasks;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

pub use layout::{Canonical, Encoding, Field, Logical, Piece, Segment, StorageLayout};
pub use region::Span;
pub use tasks::Spans;

use crate::Key;
use crate::protocol::LinkBandwidth;

/// Why a manifest cannot be computed or applied.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A layout or request contradicts itself.
    #[error("invalid copy manifest input: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A SHA-256 digest of length-prefixed parts under a domain tag.
pub(crate) struct Digest(Sha256);

impl Digest {
    pub(crate) fn new(tag: &[u8]) -> Self {
        let mut digest = Self(Sha256::new());
        digest.bytes(tag);
        digest
    }

    pub(crate) fn bytes(&mut self, part: &[u8]) {
        self.0.update((part.len() as u64).to_le_bytes());
        self.0.update(part);
    }

    pub(crate) fn text(&mut self, part: &str) {
        self.bytes(part.as_bytes());
    }

    pub(crate) fn number(&mut self, value: u64) {
        self.0.update(value.to_le_bytes());
    }

    pub(crate) fn numbers(&mut self, values: &[u64]) {
        self.number(values.len() as u64);
        for value in values {
            self.number(*value);
        }
    }

    pub(crate) fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

/// One chunk of a storage: its bytes and the key its memory is retained
/// under.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Chunk {
    pub offset: u64,
    pub length: u64,
    pub key: Key,
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
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layout, chunks })
    }
}

/// A GPU, by its UUID, as the Storage Server and the Transfer Server name
/// it.
pub type Gpu = [u8; 16];

/// A GPU's UUID in its usual text form.
pub(crate) fn uuid(gpu: &Gpu) -> String {
    let hex: String = gpu.iter().map(|byte| format!("{byte:02x}")).collect();
    format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    )
}

/// One rank of a world.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rank {
    /// The GPU the rank's storages lie on; `None` for a target rank, which
    /// the manifest assigns a GPU.
    pub gpu: Option<Gpu>,
    pub storages: Vec<Storage>,
}

/// The ranks of one world under one parallel configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct World {
    pub ranks: Vec<Rank>,
}

/// What a manifest is computed from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub source: World,
    pub target: World,
    /// The GPUs the target world's ranks may take, at least one per rank.
    pub gpus: Vec<Gpu>,
    /// The bandwidth of every link between them, as the host's Transfer
    /// Server measured it (the listing's `transfer.links`): when it started
    /// for the links within its host, at their first use for the links from
    /// another host's GPUs (over the NVLink fabric, or READ). A copy within
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
    pub source: Gpu,
    pub target: Gpu,
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

    /// Whether the manifest is the one `digest` names: what a worker checks
    /// before executing any of it.
    pub fn verify(&self, digest: &[u8; 32]) -> bool {
        self.digest == *digest && self.compute_digest().is_ok_and(|own| own == *digest)
    }

    /// The bytes the copies move.
    pub fn copied_bytes(&self) -> u64 {
        self.copies.iter().map(Copy::bytes).sum()
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
