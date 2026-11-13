//! Exclusive storage: device storage no other owner of the rank's scope uses, as its owner takes
//! it.
//!
//! An owner whose contents outlive the engine's process takes its storage through
//! [`ExclusiveStorage`] instead of allocating it: the cache planes, which are the memory of a
//! buffer pool region. Two things implement the interface. The rank's
//! [`StorageClient`](crate::memory::client::StorageClient) takes each storage at the host's
//! Storage Server, keyed by the region's memory and what fixes its shape, so every epoch that
//! names the region, and a restarted engine of the same scope, maps the same storage; every
//! one of them maps it writable, and the buffer pool keeps the blocks they write apart.
//! [`LocalStorage`] allocates the same storages in the process, where they end with it: the
//! form of a process that serves no deployment, such as a test or a tool.
//!
//! A segmented storage is one address range whose parts are retained apart, so part of it is
//! given back and created again while every address stays valid: the value and state planes of a
//! KV cache, which a sleep gives back and a wake restores.

use std::sync::Arc;

use platform::DeviceId;
use storage::Key;

use crate::memory::allocation::Storage;
use crate::worker::{Error, Result};

/// One exclusive device storage an owner takes: its name, its device, its length and the epoch
/// of its region's memory, which key it at the Storage Server.
#[derive(Debug, Clone, Copy)]
pub struct ExclusiveConfig<'a> {
    /// The storage's name within the rank's scope, such as `prefix_cache.key.values`.
    pub name: &'a str,
    /// The device the storage is on.
    pub device: DeviceId,
    /// The storage's length in bytes.
    pub bytes: usize,
    /// Other local devices that address the storage.
    pub access: &'a [DeviceId],
    /// The epoch whose ranks created the memory of the region the storage belongs to, as the
    /// region's segment table states it (the epoch of its first segment): it keys the storage,
    /// so every epoch that maps the region maps the same storage. `None` for a storage of no
    /// segmented region, which the rank's own epoch keys.
    pub epoch: Option<u64>,
}

/// One exclusive device storage of segments, each part retained on its own
/// ([`Storage::segmented`]).
#[derive(Debug, Clone, Copy)]
pub struct SegmentedConfig<'a> {
    /// The storage's name within the rank's scope; the part of segment `s` that column `c`
    /// holds is retained as `{name}#{c}/{s.ordinal}` of `s.epoch`.
    pub name: &'a str,
    /// The device the storage is on.
    pub device: DeviceId,
    /// The storage's parts in address order.
    pub parts: &'a [SegmentPart],
    /// Other local devices that address the storage.
    pub access: &'a [DeviceId],
}

impl SegmentedConfig<'_> {
    /// Each part's length, in address order.
    pub(crate) fn lengths(&self) -> Vec<usize> {
        self.parts.iter().map(|part| part.length).collect()
    }
}

/// One part of a segmented storage: what one column of a cache plane holds of one segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentPart {
    /// Bytes, a multiple of the allocation granularity.
    pub length: usize,
    /// The plane's column the part belongs to.
    pub column: u32,
    /// The segment: the epoch that created its storage and its ordinal there.
    pub segment: crate::memory::cache::info::CacheSegment,
    /// Whether the part is created when its storage does not exist; one that is not is left
    /// unmapped, as a segment given back.
    pub create: bool,
}

/// Where an owner of exclusive device storage takes it.
pub trait ExclusiveStorage {
    /// One storage per config, in order, mapped read-write, each with whether it was created
    /// for the owner.
    ///
    /// A created storage's contents are uninitialized, and its owner resets them. One that
    /// exists holds the finite values last written to it, which the owner keeps: another epoch
    /// that names the region may serve from them, and a restarted engine reads the pages they
    /// hold.
    fn exclusive(&self, configs: &[ExclusiveConfig<'_>]) -> Result<Vec<(Arc<Storage>, bool)>>;

    /// The key the Storage Server retains the storage of `config` under and the UUID of its
    /// GPU, which a copy between processes names it by; `None` for storage local to the
    /// process.
    fn key(&self, config: &ExclusiveConfig<'_>) -> Result<Option<(Key, [u8; 16])>>;

    /// The segmented storage of `config`, with, for each part, whether it was created for the
    /// owner (its contents are uninitialized).
    ///
    /// A part left unmapped ([`Storage::segment_mapped`]) is one whose `create` is unset and
    /// whose storage does not exist.
    fn segmented(&self, config: &SegmentedConfig<'_>) -> Result<(Arc<Storage>, Vec<bool>)>;

    /// The key the Storage Server retains part `index` of `config` under and the UUID of its
    /// GPU, whether the part is mapped or not; `None` for storage local to the process.
    ///
    /// # Errors
    ///
    /// When `config` has no part `index`.
    fn segment_key(
        &self,
        config: &SegmentedConfig<'_>,
        index: usize,
    ) -> Result<Option<(Key, [u8; 16])>>;

    /// Gives back the storage of the parts `segments` of the storage taken for `config`: they
    /// are unmapped here at once, and their storage is freed once no other process maps it.
    /// The caller has retired every device access to them.
    fn release_segments(
        &self,
        config: &SegmentedConfig<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()>;

    /// Creates the parts `segments` of the storage taken for `config` again, at their
    /// addresses; their contents are uninitialized.
    fn restore_segments(
        &self,
        config: &SegmentedConfig<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()>;
}

/// Exclusive storage a process allocates for itself, which ends with it.
pub struct LocalStorage;

impl ExclusiveStorage for LocalStorage {
    fn exclusive(&self, configs: &[ExclusiveConfig<'_>]) -> Result<Vec<(Arc<Storage>, bool)>> {
        configs
            .iter()
            .map(|config| {
                let storage = Storage::exportable(config.device, config.bytes, config.access)?;
                Ok((storage, true))
            })
            .collect()
    }

    fn key(&self, _config: &ExclusiveConfig<'_>) -> Result<Option<(Key, [u8; 16])>> {
        Ok(None)
    }

    fn segmented(&self, config: &SegmentedConfig<'_>) -> Result<(Arc<Storage>, Vec<bool>)> {
        let storage = Storage::segmented(config.device, &config.lengths(), config.access)?;
        for (index, part) in config.parts.iter().enumerate() {
            if part.create {
                storage.create_segment(index)?;
            }
        }

        let created = config.parts.iter().map(|part| part.create).collect();
        Ok((storage, created))
    }

    fn segment_key(
        &self,
        config: &SegmentedConfig<'_>,
        index: usize,
    ) -> Result<Option<(Key, [u8; 16])>> {
        if index >= config.parts.len() {
            return Err(Error::Invalid(format!(
                "{} has no part {index}",
                config.name
            )));
        }
        Ok(None)
    }

    fn release_segments(
        &self,
        _config: &SegmentedConfig<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()> {
        for index in segments {
            storage.release_segment(*index)?;
        }
        Ok(())
    }

    fn restore_segments(
        &self,
        _config: &SegmentedConfig<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()> {
        for index in segments {
            storage.create_segment(*index)?;
        }
        Ok(())
    }
}
