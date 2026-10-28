//! A manifest's copies as the spans of a transfer between the storage the
//! Storage Server retains ([`crate::protocol::Request::Transfer`]).
//!
//! The Transfer Server knows storage only as regions the Storage Server
//! holds: every chunk is retained under its key in its bound host or device storage,
//! and each staging space of a manifest is retained under
//! [`Manifest::staging_key`] on its rank's GPU, so a span names the copy of
//! a key at its physical storage ([`Region::Tensor`]), wrapped in
//! [`Region::Remote`] when another host owns it. A strided copy is one span of rows
//! at its pitches for each run of rows that lies in one chunk on each side;
//! a row that crosses a chunk bound on either side splits there into spans
//! of one row. The submitter keeps the sources unchanged and the
//! destinations unread until the transfer ends.

use super::{Binding, Copy, Endpoint, Error, Gpu, Manifest, Request, Result, Side, Space, World};
use crate::Key;
use crate::protocol::{Placement, Region, Span};

impl Manifest {
    /// The key a staging space of the manifest is retained under.
    pub fn staging_key(&self, staging: u32) -> Key {
        let mut digest = crate::key::Digest::new(b"uniserve-manifest-staging");
        digest.bytes(&self.digest);
        digest.number(u64::from(staging));
        Key(digest.finish())
    }

    /// Bind this plan to the source allocations visible to the submitting
    /// host. `sources` must describe the same ranks, layouts and retained
    /// chunks as `request.source`, in the same order; only their transport
    /// bindings may differ. GPU assignments and numerical copies stay fixed.
    pub fn bind<'m>(&'m self, request: &'m Request, sources: &'m World) -> BoundManifest<'m> {
        BoundManifest {
            manifest: self,
            request,
            sources,
        }
    }
}

/// An executable plan viewed from one submitting host. Source locations
/// are resolved here once; consumers submit the resulting regions directly.
#[derive(Clone, Copy)]
pub struct BoundManifest<'m> {
    manifest: &'m Manifest,
    request: &'m Request,
    sources: &'m World,
}

impl<'m> BoundManifest<'m> {
    /// Copies as Transfer Server spans, after source conversions and
    /// before target conversions. Both allocations remain retained until
    /// their transfer completes.
    pub fn spans(self) -> Spans<'m> {
        Spans {
            bound: self,
            copy: 0,
            row: 0,
            done: 0,
        }
    }

    /// The region holding the byte at `offset` of `endpoint` of a rank of
    /// `side`, the offset in it, and the bytes the region holds from there
    /// on.
    fn locate(&self, side: Side, endpoint: &Endpoint, offset: u64) -> Result<(Region, u64, u64)> {
        let (world, gpu): (&World, Option<Gpu>) = match side {
            Side::Source => (
                self.sources,
                self.sources
                    .ranks
                    .get(endpoint.rank as usize)
                    .and_then(|rank| rank.gpu),
            ),
            Side::Target => (
                &self.request.target,
                self.manifest.gpus.get(endpoint.rank as usize).copied(),
            ),
        };
        match endpoint.space {
            Space::Staging { staging } => {
                let gpu = gpu
                    .ok_or_else(|| Error::Invalid(format!("rank {} has no GPU", endpoint.rank)))?;
                let bytes = self
                    .manifest
                    .stagings
                    .get(staging as usize)
                    .map(|staging| staging.bytes)
                    .ok_or_else(|| Error::Invalid(format!("staging {staging} does not exist")))?;
                Ok((
                    Region::Tensor {
                        key: self.manifest.staging_key(staging),
                        at: Some(Placement::Device { gpu }),
                    },
                    offset,
                    bytes.saturating_sub(offset),
                ))
            }
            Space::Storage { storage } => {
                let chunks = &world
                    .ranks
                    .get(endpoint.rank as usize)
                    .and_then(|rank| rank.storages.get(storage as usize))
                    .ok_or_else(|| Error::Invalid(format!("storage {storage} does not exist")))?
                    .chunks;
                let index = chunks.partition_point(|chunk| chunk.offset + chunk.length <= offset);
                let chunk = chunks
                    .get(index)
                    .filter(|chunk| chunk.offset <= offset)
                    .ok_or_else(|| {
                        Error::Invalid(format!("byte {offset} of storage {storage} is in no chunk"))
                    })?;
                Ok((
                    chunk.binding.region(chunk.key, gpu)?,
                    chunk.storage_offset + offset - chunk.offset,
                    chunk.offset + chunk.length - offset,
                ))
            }
        }
    }
}

impl Binding {
    /// Bind a retained allocation to the region consumed by the Transfer
    /// Server. Remote regions are READ sources; targets must be local.
    pub fn region(&self, key: Key, gpu: Option<Gpu>) -> Result<Region> {
        let at = match self {
            Self::Rank => Placement::Device {
                gpu: gpu.ok_or_else(|| Error::Invalid("rank storage names no GPU".into()))?,
            },
            Self::Local { at } | Self::Remote { at, .. } => *at,
        };
        let region = Region::Tensor { key, at: Some(at) };
        Ok(match self {
            Self::Remote { endpoint, .. } => Region::Remote {
                endpoint: endpoint.clone(),
                region: Box::new(region),
            },
            _ => region,
        })
    }
}

/// The spans of a bound manifest's copies ([`BoundManifest::spans`]).
pub struct Spans<'m> {
    bound: BoundManifest<'m>,
    /// The copy, its row and the bytes of the row done.
    copy: usize,
    row: u64,
    done: u64,
}

impl Spans<'_> {
    /// The next span of `copy`, from row `self.row` and byte `self.done` of
    /// it: as many whole rows as lie in one chunk on each side, or the part
    /// of a row up to the nearer chunk bound.
    fn span(&self, copy: &Copy) -> Result<Span> {
        let source_offset = copy.source.offset + self.row * copy.source_pitch + self.done;
        let target_offset = copy.target.offset + self.row * copy.target_pitch + self.done;
        let (source, source_at, source_left) =
            self.bound
                .locate(Side::Source, &copy.source, source_offset)?;
        let (destination, destination_at, destination_left) =
            self.bound
                .locate(Side::Target, &copy.target, target_offset)?;
        if self.done == 0 && copy.length <= source_left && copy.length <= destination_left {
            // Whole rows: row `r` from here lies in both chunks while
            // `r * pitch + length` fits what each holds from here.
            let left = copy.rows - self.row;
            let fitting = |held: u64, pitch: u64| match pitch {
                0 => left,
                pitch => ((held - copy.length) / pitch + 1).min(left),
            };
            let rows = fitting(source_left, copy.source_pitch)
                .min(fitting(destination_left, copy.target_pitch));
            return Ok(Span {
                source,
                source_offset: source_at,
                destination,
                destination_offset: destination_at,
                bytes: copy.length,
                rows,
                source_pitch: if rows > 1 { copy.source_pitch } else { 0 },
                destination_pitch: if rows > 1 { copy.target_pitch } else { 0 },
            });
        }
        let bytes = (copy.length - self.done)
            .min(source_left)
            .min(destination_left);
        if bytes == 0 {
            return Err(Error::Invalid("a copy reaches past its region".into()));
        }
        Ok(Span {
            source,
            source_offset: source_at,
            destination,
            destination_offset: destination_at,
            bytes,
            rows: 1,
            source_pitch: 0,
            destination_pitch: 0,
        })
    }
}

impl Iterator for Spans<'_> {
    type Item = Result<Span>;

    fn next(&mut self) -> Option<Self::Item> {
        let copy = *self.bound.manifest.copies.get(self.copy)?;
        let span = self.span(&copy);
        match &span {
            // Whole rows advance the row; a part of a row advances within
            // it.
            Ok(span) if self.done == 0 && span.bytes == copy.length => {
                self.row += span.rows;
            }
            Ok(span) => {
                self.done += span.bytes;
                if self.done == copy.length {
                    self.done = 0;
                    self.row += 1;
                }
            }
            // A failed span ends the iteration after reporting it.
            Err(_) => {
                self.copy = self.bound.manifest.copies.len();
                return Some(span);
            }
        }
        if self.row == copy.rows {
            self.row = 0;
            self.copy += 1;
        }
        Some(span)
    }
}
