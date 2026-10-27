//! A manifest's copies as the Transfer Server's tasks: byte ranges of the
//! memory regions the Storage Server registered.
//!
//! The Transfer Server knows memory only as registered regions: every chunk
//! of a storage is retained under its key on its GPU, and each staging
//! space of a manifest is registered under [`Manifest::staging_key`] on
//! its rank's GPU. A task copies `length` bytes from one region's offset to
//! another's; a copy whose bytes cross chunks on either side splits at
//! every chunk bound, and a strided copy yields one task per row.

use serde::{Deserialize, Serialize};

use super::{Copy, Endpoint, Error, Manifest, Request, Result, Side, Space, World};
use crate::Key;

/// A memory region the Storage Server registered: a storage chunk or a
/// staging space, by its key, on a GPU.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Region {
    pub key: Key,
    pub gpu: String,
}

/// One transfer: `length` bytes from `source` at `source_offset` to
/// `target` at `target_offset`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CopyTask {
    pub source: Region,
    pub source_offset: u64,
    pub target: Region,
    pub target_offset: u64,
    pub length: u64,
    /// The Transfer Server's priority, higher first.
    pub priority: u8,
}

impl Manifest {
    /// The key a staging space of the manifest is registered under.
    pub fn staging_key(&self, staging: u32) -> Key {
        let mut digest = super::Digest::new(b"uniserve-manifest-staging");
        digest.bytes(&self.digest);
        digest.number(u64::from(staging));
        Key(digest.finish())
    }

    /// The tasks that execute the copies of a manifest computed from
    /// `request`, every one at `priority`, copy by copy and row by row.
    pub fn tasks<'m>(&'m self, request: &'m Request, priority: u8) -> Tasks<'m> {
        Tasks {
            manifest: self,
            request,
            priority,
            copy: 0,
            row: 0,
            done: 0,
        }
    }

    /// The region and the offset in it of the byte at `endpoint` of a rank
    /// of `side`, and the bytes the region holds from there on.
    fn locate(
        &self,
        request: &Request,
        side: Side,
        endpoint: &Endpoint,
        offset: u64,
    ) -> Result<(Region, u64, u64)> {
        let (world, gpu): (&World, Option<&str>) = match side {
            Side::Source => (
                &request.source,
                request
                    .source
                    .ranks
                    .get(endpoint.rank as usize)
                    .and_then(|rank| rank.gpu.as_deref()),
            ),
            Side::Target => (
                &request.target,
                self.gpus.get(endpoint.rank as usize).map(String::as_str),
            ),
        };
        let gpu = gpu
            .ok_or_else(|| Error::Invalid(format!("rank {} has no GPU", endpoint.rank)))?
            .to_owned();
        match endpoint.space {
            Space::Staging { staging } => {
                let bytes = self
                    .stagings
                    .get(staging as usize)
                    .map(|staging| staging.bytes)
                    .ok_or_else(|| Error::Invalid(format!("staging {staging} does not exist")))?;
                Ok((
                    Region {
                        key: self.staging_key(staging),
                        gpu,
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
                    Region {
                        key: chunk.key,
                        gpu,
                    },
                    offset - chunk.offset,
                    chunk.offset + chunk.length - offset,
                ))
            }
        }
    }
}

/// The tasks of a manifest's copies ([`Manifest::tasks`]).
pub struct Tasks<'m> {
    manifest: &'m Manifest,
    request: &'m Request,
    priority: u8,
    /// The copy, its row and the bytes of the row done.
    copy: usize,
    row: u64,
    done: u64,
}

impl Tasks<'_> {
    fn task(&self, copy: &Copy) -> Result<CopyTask> {
        let source_offset = copy.source.offset + self.row * copy.source_pitch + self.done;
        let target_offset = copy.target.offset + self.row * copy.target_pitch + self.done;
        let (source, source_at, source_left) =
            self.manifest
                .locate(self.request, Side::Source, &copy.source, source_offset)?;
        let (target, target_at, target_left) =
            self.manifest
                .locate(self.request, Side::Target, &copy.target, target_offset)?;
        let length = (copy.length - self.done).min(source_left).min(target_left);
        if length == 0 {
            return Err(Error::Invalid("a copy reaches past its region".into()));
        }
        Ok(CopyTask {
            source,
            source_offset: source_at,
            target,
            target_offset: target_at,
            length,
            priority: self.priority,
        })
    }
}

impl Iterator for Tasks<'_> {
    type Item = Result<CopyTask>;

    fn next(&mut self) -> Option<Self::Item> {
        let copy = *self.manifest.copies.get(self.copy)?;
        let task = self.task(&copy);
        match &task {
            Ok(task) => {
                self.done += task.length;
                if self.done == copy.length {
                    self.done = 0;
                    self.row += 1;
                    if self.row == copy.rows {
                        self.row = 0;
                        self.copy += 1;
                    }
                }
            }
            // A failed task ends the iteration after reporting it.
            Err(_) => self.copy = self.manifest.copies.len(),
        }
        Some(task)
    }
}
