//! A manifest's copies as the spans of a transfer between the memory the
//! Storage Server retains ([`crate::protocol::Request::Transfer`]).
//!
//! The Transfer Server knows memory only as regions the Storage Server
//! holds: every chunk of a storage is retained under its key on its GPU,
//! and each staging space of a manifest is retained under
//! [`Manifest::staging_key`] on its rank's GPU, so a span names the copy of
//! a key at a GPU ([`Region::Tensor`]). A copy whose bytes cross chunks on
//! either side splits at every chunk bound, and a strided copy yields one
//! span per row. The submitter keeps the sources unchanged and the
//! destinations unread until the transfer ends.

use super::{Copy, Endpoint, Error, Gpu, Manifest, Request, Result, Side, Space, World};
use crate::Key;
use crate::protocol::{Placement, Region, Span};

impl Manifest {
    /// The key a staging space of the manifest is retained under.
    pub fn staging_key(&self, staging: u32) -> Key {
        let mut digest = super::Digest::new(b"uniserve-manifest-staging");
        digest.bytes(&self.digest);
        digest.number(u64::from(staging));
        Key(digest.finish())
    }

    /// The spans that execute the copies of a manifest computed from
    /// `request`, copy by copy and row by row: submitted as transfers
    /// after the source conversions ran, and followed by the target
    /// conversions.
    pub fn spans<'m>(&'m self, request: &'m Request) -> Spans<'m> {
        Spans {
            manifest: self,
            request,
            copy: 0,
            row: 0,
            done: 0,
        }
    }

    /// The region holding the byte at `offset` of `endpoint` of a rank of
    /// `side`, the offset in it, and the bytes the region holds from there
    /// on.
    fn locate(
        &self,
        request: &Request,
        side: Side,
        endpoint: &Endpoint,
        offset: u64,
    ) -> Result<(Region, u64, u64)> {
        let (world, gpu): (&World, Option<Gpu>) = match side {
            Side::Source => (
                &request.source,
                request
                    .source
                    .ranks
                    .get(endpoint.rank as usize)
                    .and_then(|rank| rank.gpu),
            ),
            Side::Target => (
                &request.target,
                self.gpus.get(endpoint.rank as usize).copied(),
            ),
        };
        let gpu =
            gpu.ok_or_else(|| Error::Invalid(format!("rank {} has no GPU", endpoint.rank)))?;
        let at = Some(Placement::Device { gpu });
        match endpoint.space {
            Space::Staging { staging } => {
                let bytes = self
                    .stagings
                    .get(staging as usize)
                    .map(|staging| staging.bytes)
                    .ok_or_else(|| Error::Invalid(format!("staging {staging} does not exist")))?;
                Ok((
                    Region::Tensor {
                        key: self.staging_key(staging),
                        at,
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
                    Region::Tensor { key: chunk.key, at },
                    offset - chunk.offset,
                    chunk.offset + chunk.length - offset,
                ))
            }
        }
    }
}

/// The spans of a manifest's copies ([`Manifest::spans`]).
pub struct Spans<'m> {
    manifest: &'m Manifest,
    request: &'m Request,
    /// The copy, its row and the bytes of the row done.
    copy: usize,
    row: u64,
    done: u64,
}

impl Spans<'_> {
    fn span(&self, copy: &Copy) -> Result<Span> {
        let source_offset = copy.source.offset + self.row * copy.source_pitch + self.done;
        let target_offset = copy.target.offset + self.row * copy.target_pitch + self.done;
        let (source, source_at, source_left) =
            self.manifest
                .locate(self.request, Side::Source, &copy.source, source_offset)?;
        let (destination, destination_at, destination_left) =
            self.manifest
                .locate(self.request, Side::Target, &copy.target, target_offset)?;
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
        })
    }
}

impl Iterator for Spans<'_> {
    type Item = Result<Span>;

    fn next(&mut self) -> Option<Self::Item> {
        let copy = *self.manifest.copies.get(self.copy)?;
        let span = self.span(&copy);
        match &span {
            Ok(span) => {
                self.done += span.bytes;
                if self.done == copy.length {
                    self.done = 0;
                    self.row += 1;
                    if self.row == copy.rows {
                        self.row = 0;
                        self.copy += 1;
                    }
                }
            }
            // A failed span ends the iteration after reporting it.
            Err(_) => self.copy = self.manifest.copies.len(),
        }
        Some(span)
    }
}
