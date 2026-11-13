//! The cache's planes: the device storage of the unit pool and its unit-level resets and copies.
//!
//! Each field (keys, values) owns separate storages, cell-major, a cell being one column of one
//! unit at `column · units + unit`:
//!
//! | Storage | Element | Shape |
//! | --- | --- | --- |
//! | `prefix_cache.{key,value}.values` | bytes | `[columns, units, plane_bytes]` |
//! | `prefix_cache.{key,value}.scale` (FP8 only) | F32 | `[columns, units]` |
//! | `prefix_cache.{key,value}.initialized` | U8 | `[columns, units]` |
//! | `prefix_cache.index_key.values` (index keys only) | bytes | `[columns, units, index_bytes]` |
//! | `prefix_cache.state.{run}.values` (state group only) | bytes | `[columns, units, plane_bytes]` |
//!
//! The names key the storages at the Storage Server, where every epoch that names the cache's
//! region and a restarted engine find them, so they stay as they are whatever this module is
//! called. Every storage of the planes is the memory of the region: the value planes' segments are
//! keyed by the segments the region's segment table names, and the other storages by the epoch
//! of its first segment, the epoch whose ranks created the region's memory. Two epochs of one
//! layout therefore address the same cells, and the worker that is given a unit resets its
//! flags and scales before a batch writes it ([`Planes::reset_units`]).
//!
//! An attention group's page is one unit and each of its layers one column, so a layer's keys
//! are the contiguous `[units, block_size, kv_heads, head_dim]` of its column and a page id is
//! a unit id. A draft layer lies over `span` columns, so its pages are the `span · units`
//! planes of those columns. A layer's index keys are its column's cells of the index-key
//! storage in the same way, `[units, block_size / ratio, width]`, so a page's keys, values and
//! index keys are addressed by one page id.
//!
//! On a device whose cache splits ([`super::segments`]), each value storage is segmented: the
//! last segments are given back ([`Planes::release_segments`]) and created again at their
//! addresses ([`Planes::restore_segments`]). The index keys are not segmented: like the flags,
//! the scales and the state planes they are one storage that stays mapped.

use std::sync::Arc;

use nn::KvPlanes;
use platform::{CopyBatch, DeviceId, SourceOrder, Stream, memset_2d_d8, memset_2d_d32};
use tensor::{DType, Tensor};

use super::layout::{CacheLayout, GroupLayout};
use super::segments::{self, PlaneSegment};
use crate::memory::allocation::Storage;
use crate::memory::exclusive::{ExclusiveConfig, ExclusiveStorage, SegmentPart, SegmentedConfig};
use crate::memory::lend;
use crate::worker::{Error, Result};

/// The value storages' names, keys then values.
const VALUES: [&str; 2] = ["prefix_cache.key.values", "prefix_cache.value.values"];

/// The initialization flag storages' names, keys then values.
const FLAGS: [&str; 2] = [
    "prefix_cache.key.initialized",
    "prefix_cache.value.initialized",
];

/// The FP8 scale storages' names, keys then values.
const SCALES: [&str; 2] = ["prefix_cache.key.scale", "prefix_cache.value.scale"];

/// The index-key storage's name.
const INDEX_KEYS: &str = "prefix_cache.index_key.values";

/// The bit pattern of `1.0f32`, the neutral FP8 scale.
const UNIT_SCALE: u32 = 0x3F80_0000;

/// The consecutive runs of `units`, as `(first, count)` in ascending order, each unit once.
pub(crate) fn unit_runs(units: &[u32]) -> Vec<(u32, u32)> {
    let mut units = units.to_vec();
    units.sort_unstable();
    units.dedup();

    let mut runs: Vec<(u32, u32)> = Vec::new();
    for unit in units {
        match runs.last_mut() {
            Some((first, count)) if *first + *count == unit => *count += 1,
            _ => runs.push((unit, 1)),
        }
    }
    runs
}

/// The segments of a device cache's value planes and which of them are mapped.
struct Segments {
    /// The segments in unit order.
    segments: Vec<crate::memory::cache::info::CacheSegment>,
    /// The first segment given back; the segment count while none is.
    released_from: u32,
    /// The cache's columns, which `parts` is laid out by.
    columns: u32,
    /// Each value plane's parts, column-major: the part of segment `k` that column `c` holds
    /// is `c · segments.len() + k`.
    parts: Vec<SegmentPart>,
}

impl Segments {
    fn count(&self) -> u32 {
        self.segments.len() as u32
    }

    /// The first unit of segment `index`.
    fn first_unit(&self, index: u32) -> u32 {
        self.segments
            .iter()
            .take(index as usize)
            .map(|segment| segment.units)
            .sum()
    }

    /// The parts of segments `range`, every column's.
    fn parts_of(&self, range: std::ops::Range<u32>) -> Vec<usize> {
        let count = self.count();

        (0..self.columns)
            .flat_map(|column| {
                range
                    .clone()
                    .map(move |segment| (column * count + segment) as usize)
            })
            .collect()
    }
}

/// The storages of a cache on a device.
struct DevicePlanes {
    device: DeviceId,
    /// Other local devices that address the planes: readers there copy pages directly.
    peers: Vec<DeviceId>,
    values: [Arc<Storage>; 2],
    initialized: [Arc<Storage>; 2],
    /// The per-plane scales of an FP8 cache.
    scales: Option<[Arc<Storage>; 2]>,
    /// The index-key planes of a cache whose layers keep index keys.
    index_keys: Option<Arc<Storage>>,
    /// The state group's checkpoint planes, one storage per field run of a state page.
    state: Vec<Arc<Storage>>,
    /// The value planes' segments, when the cache splits into them.
    segments: Option<Segments>,
    /// Where the Storage Server retains every unit's bytes, run by run.
    runs: Vec<crate::memory::cache::info::CachePlaneRun>,
}

/// The unit pool's storage: on one device, or none for a cache that holds no device, whose
/// tensors are meta tensors.
pub(crate) struct Planes {
    layout: Arc<CacheLayout>,
    units: u32,
    device: Option<DevicePlanes>,
}

impl Planes {
    /// Planes of `units` units that hold no storage: every tensor is a meta tensor.
    pub(crate) fn meta(layout: Arc<CacheLayout>, units: u32) -> Result<Self> {
        check(&layout, units)?;

        Ok(Self {
            layout,
            units,
            device: None,
        })
    }

    /// Takes the planes of `units` units (the padding unit included) on `device` from `owner`,
    /// the value planes in `parts` segments when the cache splits, and resets what was created
    /// for this rank on `stream`, which is synchronized before returning: values, flags and
    /// index keys zero, scales one.
    ///
    /// Planes that exist keep their contents, which are finite whatever they hold: another
    /// epoch that names the region serves from them, a restarted engine reads the pages its
    /// predecessor published, and every unit is reset for its next owner. Every storage is
    /// keyed by the region's memory (module documentation): the storages that are not split by
    /// the epoch of the first of `parts`, and by the rank's own epoch when the planes do not
    /// split.
    ///
    /// # Errors
    ///
    /// When the segments do not hold `units` units, the segments of one plane are not given
    /// back from one segment on in every column, `owner` refuses a storage, or a reset fails.
    pub(crate) fn open(
        layout: Arc<CacheLayout>,
        units: u32,
        parts: Option<&[PlaneSegment]>,
        device: DeviceId,
        owner: &dyn ExclusiveStorage,
        stream: &Stream,
    ) -> Result<Self> {
        check(&layout, units)?;
        if parts.is_some_and(|parts| segments::units(parts) != units) {
            return Err(Error::Invalid(
                "a cache's segments do not hold its units".into(),
            ));
        }

        let peers = Storage::peers(device)?;
        let cells = layout.columns as usize * units as usize;
        let plane_bytes = cells * layout.plane_bytes as usize;
        let index_bytes = cells * layout.index_bytes() as usize;

        // The epoch whose ranks created the region's memory: that of its first segment, which
        // holds unit 0. A segment given back and created again keeps its identity, so the epoch
        // stays the same for the region's life.
        let epoch = parts
            .and_then(|parts| parts.first())
            .map(|part| part.segment.epoch);
        let config = |name, bytes| ExclusiveConfig {
            name,
            device,
            bytes,
            access: &peers,
            epoch,
        };

        // The exclusive storages, in the order they are taken apart below: the value planes of
        // a cache that does not split, the flags, the scales, the index keys, the state runs.
        let state_names: Vec<String> = (0..layout.state_fields())
            .map(|run| format!("prefix_cache.state.{run}.values"))
            .collect();
        let mut configs = Vec::new();
        if parts.is_none() {
            configs.extend(VALUES.map(|name| config(name, plane_bytes)));
        }
        configs.extend(FLAGS.map(|name| config(name, cells)));
        if layout.quantized {
            configs.extend(SCALES.map(|name| config(name, cells * 4)));
        }
        if index_bytes > 0 {
            configs.push(config(INDEX_KEYS, index_bytes));
        }
        configs.extend(state_names.iter().map(|name| config(name, plane_bytes)));

        let mut runs = Vec::new();
        for config in &configs {
            let cell_bytes = (config.bytes / cells) as u64;
            runs.extend(cell_runs(owner, config, layout.columns, units, cell_bytes)?);
        }

        let mut taken = owner.exclusive(&configs)?.into_iter();
        let mut pair = || -> Result<([Arc<Storage>; 2], bool)> {
            match (taken.next(), taken.next()) {
                (Some((key, key_created)), Some((value, value_created))) => {
                    Ok(([key, value], key_created || value_created))
                }
                _ => Err(Error::Invariant("an exclusive storage is missing".into())),
            }
        };

        // Whether the values are new as a whole, and the segments created beside ones that
        // exist, whose units are reset alone.
        let values_created;
        let mut created_segments: Vec<u32> = Vec::new();
        let mut segmented = None;

        let values = match parts {
            None => {
                let (values, created) = pair()?;
                values_created = created;
                values
            }
            Some(parts) => {
                let (values, segments, created) =
                    take_segmented(&layout, parts, device, &peers, owner, &mut runs)?;
                values_created = created.len() == parts.len();
                if !values_created {
                    created_segments = created;
                }
                segmented = Some(segments);
                values
            }
        };
        let (initialized, flags_created) = pair()?;
        let (scales, scales_created) = if layout.quantized {
            let (scales, created) = pair()?;
            (Some(scales), created)
        } else {
            (None, false)
        };
        let (index_keys, index_keys_created) = if index_bytes > 0 {
            let (storage, created) = taken
                .next()
                .ok_or_else(|| Error::Invariant("the index-key storage is missing".into()))?;
            (Some(storage), created)
        } else {
            (None, false)
        };

        let mut state = Vec::with_capacity(state_names.len());
        let mut state_created = false;
        for (storage, created) in taken {
            state_created |= created;
            state.push(storage);
        }
        if state.len() != state_names.len() {
            return Err(Error::Invariant(
                "a state plane's storage is missing".into(),
            ));
        }

        let planes = Self {
            layout,
            units,
            device: Some(DevicePlanes {
                device,
                peers,
                values,
                initialized,
                scales,
                index_keys,
                state,
                segments: segmented,
                runs,
            }),
        };
        let held = planes.device()?;

        for field in 0..2 {
            // SAFETY: each fill spans exactly one storage this value owns, whose segments are
            // all mapped when its values were created as a whole; the stream is drained below.
            unsafe {
                if values_created {
                    fill(stream, held.values[field].address(), 0, 1, plane_bytes)?;
                }
                if flags_created {
                    fill(stream, held.initialized[field].address(), 0, 1, cells)?;
                }
                if let Some(scales) = &held.scales
                    && scales_created
                {
                    fill(stream, scales[field].address(), UNIT_SCALE, 4, cells)?;
                }
            }
        }
        if let Some(storage) = &held.index_keys
            && index_keys_created
        {
            // SAFETY: the fill spans exactly the storage, which this value owns.
            unsafe { fill(stream, storage.address(), 0, 1, index_bytes) }?;
        }
        if state_created {
            for storage in &held.state {
                // SAFETY: the fill spans exactly the storage, which this value owns.
                unsafe { fill(stream, storage.address(), 0, 1, plane_bytes) }?;
            }
        }
        if let Some(segments) = &held.segments {
            for segment in created_segments {
                let count = segments.segments[segment as usize].units;
                planes.reset_units(segments.first_unit(segment), count, true, stream)?;
            }
        }
        stream.synchronize()?;

        Ok(planes)
    }

    /// The device side, or the refusal of planes that hold no device.
    fn device(&self) -> Result<&DevicePlanes> {
        self.device.as_ref().ok_or(Error::State(
            "the cache holds no device: its planes have no storage",
        ))
    }

    /// The device the planes are on; `None` for planes that hold none.
    pub(crate) fn device_id(&self) -> Option<DeviceId> {
        self.device.as_ref().map(|device| device.device)
    }

    /// Where the Storage Server retains every unit's bytes, run by run; empty for planes local
    /// to the process and for planes that hold no device.
    pub(crate) fn runs(&self) -> &[crate::memory::cache::info::CachePlaneRun] {
        self.device
            .as_ref()
            .map_or(&[], |device| device.runs.as_slice())
    }

    /// The segments the value planes are made of, in unit order; `None` for planes that are
    /// not split.
    pub(crate) fn segments(&self) -> Option<&[crate::memory::cache::info::CacheSegment]> {
        let segments = self.device.as_ref()?.segments.as_ref()?;
        Some(&segments.segments)
    }

    /// The segments the planes hold, `[0, held)`; zero for planes that are not split.
    pub(crate) fn held_segments(&self) -> u32 {
        self.device
            .as_ref()
            .and_then(|device| device.segments.as_ref())
            .map_or(0, |segments| segments.released_from)
    }

    /// The device bytes one unit of both value planes holds over every column: what a segment
    /// gives back per unit, which the flags, scales, index keys and state planes are no part
    /// of.
    pub(crate) fn unit_bytes(&self) -> u64 {
        2 * u64::from(self.layout.columns) * self.layout.plane_bytes
    }

    /// The bytes the planes map on the device: the value planes' mapped segments, the flags,
    /// the scales, the index keys and the state planes. A sleeping rank's segments are not in
    /// it.
    pub(crate) fn mapped_bytes(&self) -> u64 {
        let Some(device) = &self.device else {
            return 0;
        };
        let bytes = |storages: &[Arc<Storage>]| -> u64 {
            storages.iter().map(|storage| storage.bytes() as u64).sum()
        };

        // A segment is given back from both value planes, and the storages keep their length
        // while their segments are unmapped.
        let released = device.segments.as_ref().map_or(0, |segments| {
            let plane: u64 = segments
                .parts_of(segments.released_from..segments.count())
                .into_iter()
                .map(|part| segments.parts[part].length as u64)
                .sum();
            plane * device.values.len() as u64
        });

        bytes(&device.values).saturating_sub(released)
            + bytes(&device.initialized)
            + device.scales.as_ref().map_or(0, |scales| bytes(scales))
            + bytes(device.index_keys.as_slice())
            + bytes(&device.state)
    }

    fn cell(&self, column: u32, unit: u32) -> u64 {
        u64::from(column) * u64::from(self.units) + u64::from(unit)
    }

    /// Refuses a unit outside the pool.
    pub(crate) fn require_unit(&self, unit: u32) -> Result<()> {
        if unit >= self.units {
            return Err(Error::Invalid(format!(
                "cache unit {unit} is outside the pool of {} units",
                self.units
            )));
        }
        Ok(())
    }

    /// The tensor of `shape` elements of `dtype` whose first element lies `offset` bytes into
    /// `storage` of the device side, or the meta tensor of that shape.
    fn tensor(
        &self,
        storage: impl FnOnce(&DevicePlanes) -> &Arc<Storage>,
        offset: u64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        match &self.device {
            // SAFETY: the planes own the storage for as long as they live, and the holder of a
            // cache keeps it for as long as tensors over it and work issued on them exist; a
            // window over segments given back is addressed by no unit the engine allocates.
            Some(device) => unsafe { storage(device).external(offset as usize, shape, dtype) },
            // SAFETY: a meta tensor has no data and its address is never read.
            None => unsafe { lend(0, shape, dtype, None) },
        }
    }

    /// The scales of `field` from cell `cell` on, as `shape` F32 elements; `None` for a dense
    /// cache, which stores none.
    fn scales(&self, field: usize, cell: u64, shape: &[usize]) -> Result<Option<Tensor>> {
        if !self.layout.quantized {
            return Ok(None);
        }

        let scales = match &self.device {
            Some(device) => {
                let scales = device.scales.as_ref().ok_or_else(|| {
                    Error::Invariant("an FP8 cache holds no scale storage".into())
                })?;

                // SAFETY: as in `Self::tensor`: the planes own the storage.
                unsafe { scales[field].external((cell * 4) as usize, shape, DType::F32) }?
            }
            // SAFETY: a meta tensor has no data and its address is never read.
            None => unsafe { lend(0, shape, DType::F32, None) }?,
        };
        Ok(Some(scales))
    }

    /// The index keys of `group`'s layers from cell `cell` on, as `pages` pages of `tokens`
    /// tokens: `[pages, tokens / ratio, width]` in the group's compute type; `None` for a
    /// group whose layers keep none.
    fn index_keys(
        &self,
        group: &GroupLayout,
        cell: u64,
        pages: usize,
        tokens: u32,
    ) -> Result<Option<Tensor>> {
        let Some(shape) = self.layout.index_shape(group, pages, tokens)? else {
            return Ok(None);
        };
        let dtype = group.compute_dtype;

        let keys = match &self.device {
            Some(device) => {
                let storage = device.index_keys.as_ref().ok_or_else(|| {
                    Error::Invariant(
                        "a cache that keeps index keys holds no storage of them".into(),
                    )
                })?;
                let offset = cell * self.layout.index_bytes();

                // SAFETY: as in `Self::tensor`: the planes own the storage, and a page's keys
                // fill one index-key plane, so the window is the storage's cells from `cell`.
                unsafe { storage.external(offset as usize, &shape, dtype) }?
            }
            // SAFETY: a meta tensor has no data and its address is never read.
            None => unsafe { lend(0, &shape, dtype, None) }?,
        };
        Ok(Some(keys))
    }

    /// Layer `layer` of the attention group `group`: its column's keys and values as
    /// `[units, block_size, kv_heads, head_dim]`, the flags and scales of the same cells as
    /// `[units, 1, 1, 1]`, and the index keys of a group that keeps them as
    /// `[units, block_size / ratio, width]`. A page id of the group is a unit id.
    pub(crate) fn layer(&self, group: &GroupLayout, layer: usize) -> Result<KvPlanes> {
        if layer >= group.layers.len() {
            return Err(Error::Invalid(format!(
                "a cache group of {} layers holds no layer {layer}",
                group.layers.len()
            )));
        }

        // The group's page is one unit, so its layers are the cache's columns.
        self.column_planes(group, layer as u32, 1, group.block_size)
    }

    /// Layer `layer` of the draft group `group`, lying over `span` columns: pages of
    /// `block_size / span` tokens, `[span · units, block_size / span, kv_heads, head_dim]`,
    /// page `c · units + u` holding the `c`-th run of the draft page in unit `u`, with the
    /// flags, scales and index keys of the same cells.
    pub(crate) fn draft_layer(
        &self,
        group: &GroupLayout,
        layer: usize,
        span: u32,
    ) -> Result<KvPlanes> {
        if layer >= group.layers.len() {
            return Err(Error::Invalid(format!(
                "the draft group holds {} layers, not {layer}",
                group.layers.len()
            )));
        }
        self.column_planes(group, layer as u32 * span, span, group.block_size / span)
    }

    /// The planes of `columns` columns from `first`, as pages of `tokens` tokens of `group`'s
    /// rows: the columns' cells lie contiguously in each storage.
    fn column_planes(
        &self,
        group: &GroupLayout,
        first: u32,
        columns: u32,
        tokens: u32,
    ) -> Result<KvPlanes> {
        if u64::from(tokens) * group.row_bytes() != self.layout.plane_bytes {
            return Err(Error::Invariant(format!(
                "a page of {tokens} tokens of {} bytes does not fill a plane of {} bytes",
                group.row_bytes(),
                self.layout.plane_bytes
            )));
        }

        let cell = self.cell(first, 0);
        let pages = columns as usize * self.units as usize;
        let values = [
            pages,
            tokens as usize,
            group.num_kv_heads as usize,
            group.head_dim as usize,
        ];
        let cells = [pages, 1, 1, 1];
        let plane_bytes = self.layout.plane_bytes;

        let value = |field: usize| {
            self.tensor(
                |device| &device.values[field],
                cell * plane_bytes,
                &values,
                group.dtype,
            )
        };
        let flags = |field: usize| {
            self.tensor(|device| &device.initialized[field], cell, &cells, DType::U8)
        };
        let scale = |field: usize| self.scales(field, cell, &cells);

        Ok(KvPlanes {
            key: value(0)?,
            value: value(1)?,
            key_initialized: flags(0)?,
            value_initialized: flags(1)?,
            key_scale: scale(0)?,
            value_scale: scale(1)?,
            index_keys: self.index_keys(group, cell, pages, tokens)?,
        })
    }

    /// The stored values of `field` as `[columns, units, plane_bytes]` bytes, its scales as
    /// `[columns, units]` F32 for an FP8 cache, and its flags as `[columns, units]` U8: the
    /// fields `cache_copy_region` writes a converted page into.
    pub(crate) fn fields(&self, field: usize) -> Result<(Tensor, Option<Tensor>, Tensor)> {
        let (columns, units) = (self.layout.columns as usize, self.units as usize);
        let cells = [columns, units];

        let values = self.tensor(
            |device| &device.values[field],
            0,
            &[columns, units, self.layout.plane_bytes as usize],
            DType::U8,
        )?;
        let scales = self.scales(field, 0, &cells)?;
        let initialized = self.tensor(|device| &device.initialized[field], 0, &cells, DType::U8)?;

        Ok((values, scales, initialized))
    }

    /// The state group's checkpoint planes, one tensor per field run of a state page:
    /// `[columns · units, plane_bytes]` bytes, plane `(column, unit)` at row
    /// `column · units + unit`. Empty for a cache without a state group.
    pub(crate) fn state_runs(&self) -> Result<Vec<Tensor>> {
        let cells = self.layout.columns as usize * self.units as usize;
        let shape = [cells, self.layout.plane_bytes as usize];

        (0..self.layout.state_fields() as usize)
            .map(|run| match &self.device {
                Some(device) => {
                    let storage = device.state.get(run).ok_or_else(|| {
                        Error::Invariant(format!("the cache holds no state field run {run}"))
                    })?;

                    // SAFETY: as in `Self::tensor`: the planes own the storage.
                    unsafe { storage.external(0, shape.as_slice(), DType::U8) }
                }
                // SAFETY: a meta tensor has no data and its address is never read.
                None => unsafe { lend(0, shape.as_slice(), DType::U8, None) },
            })
            .collect()
    }

    /// The address of the plane of `field` in `column` of `unit`.
    pub(crate) fn plane_address(&self, field: usize, column: u32, unit: u32) -> Result<u64> {
        self.require_unit(unit)?;
        let device = self.device()?;
        Ok(device.values[field].address() + self.cell(column, unit) * self.layout.plane_bytes)
    }

    /// The address of the state plane of field run `run` in `column` of `unit`.
    pub(crate) fn state_plane_address(&self, run: usize, column: u32, unit: u32) -> Result<u64> {
        self.require_unit(unit)?;
        let device = self.device()?;
        let storage = device.state.get(run).ok_or_else(|| {
            Error::Invalid(format!(
                "a cache holds {} state field runs, not {run}",
                device.state.len()
            ))
        })?;
        Ok(storage.address() + self.cell(column, unit) * self.layout.plane_bytes)
    }

    /// The address of the FP8 scale of `field` in `column` of `unit`.
    pub(crate) fn scale_address(&self, field: usize, column: u32, unit: u32) -> Result<u64> {
        self.require_unit(unit)?;
        let scales = self
            .device()?
            .scales
            .as_ref()
            .ok_or_else(|| Error::Invalid("a dense cache's planes carry no scales".into()))?;
        Ok(scales[field].address() + self.cell(column, unit) * 4)
    }

    /// The address of the index-key plane in `column` of `unit`.
    pub(crate) fn index_address(&self, column: u32, unit: u32) -> Result<u64> {
        self.require_unit(unit)?;
        let storage = self
            .device()?
            .index_keys
            .as_ref()
            .ok_or_else(|| Error::Invalid("the cache's planes carry no index keys".into()))?;
        Ok(storage.address() + self.cell(column, unit) * self.layout.index_bytes())
    }

    /// The address of the initialization flag of `field` in `column` of `unit`.
    pub(crate) fn flag_address(&self, field: usize, column: u32, unit: u32) -> Result<u64> {
        self.require_unit(unit)?;
        Ok(self.device()?.initialized[field].address() + self.cell(column, unit))
    }

    /// Resets units `[first, first + count)` of every column with fills issued on `stream`:
    /// flags zero, scales one, and the values zero when `values`. The index keys stay as they
    /// are: they have no flag, and a unit's next owner reads no key before it stored it. The
    /// values of segmented planes fill column by column: a fill's rows may not span a segment
    /// given back, which the driver refuses although no row touches it.
    pub(crate) fn reset_units(
        &self,
        first: u32,
        count: u32,
        values: bool,
        stream: &Stream,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        self.require_unit(first + count - 1)?;
        let device = self.device()?;

        let units = self.units as usize;
        let columns = self.layout.columns as usize;
        let plane = self.layout.plane_bytes as usize;
        let (first, count) = (first as usize, count as usize);

        for field in 0..2 {
            // SAFETY: every row lies inside a storage this value owns: units
            // `[first, first + count)` of each column, a column's units apart. The caller has
            // those units mapped and orders the fills before their readers.
            unsafe {
                if values {
                    let base = device.values[field].address() + (first * plane) as u64;
                    let (rows, pitch) = (count * plane, units * plane);

                    if device.segments.is_some() {
                        for column in 0..columns {
                            fill(stream, base + (column * pitch) as u64, 0, 1, rows)?;
                        }
                    } else {
                        fill_rows(stream, base, pitch, 0, 1, rows, columns)?;
                    }
                }

                let flags = device.initialized[field].address() + first as u64;
                fill_rows(stream, flags, units, 0, 1, count, columns)?;

                if let Some(scales) = &device.scales {
                    let scales = scales[field].address() + (first * 4) as u64;
                    fill_rows(stream, scales, units * 4, UNIT_SCALE, 4, count, columns)?;
                }
            }
        }
        Ok(())
    }

    /// Raises both fields' initialization flags of `units` in every column on `stream`: the
    /// units' values and scales are in place and a reader may use them.
    pub(crate) fn mark_initialized(&self, units: &[u32], stream: &Stream) -> Result<()> {
        let device = self.device()?;
        let columns = self.layout.columns as usize;

        for (first, count) in unit_runs(units) {
            self.require_unit(first + count - 1)?;

            for initialized in &device.initialized {
                // SAFETY: the rows lie inside the flag storage this value owns, and the caller
                // orders the fill behind the copies that filled the units.
                unsafe {
                    fill_rows(
                        stream,
                        initialized.address() + u64::from(first),
                        self.units as usize,
                        1,
                        1,
                        count as usize,
                        columns,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Checks the `(source, destination)` unit copies of a block's copy: both units inside the
    /// pool, neither the padding unit, and not the same unit. A refusal names the pair and the
    /// reason.
    pub(crate) fn check_copies(&self, copies: &[(u32, u32)]) -> Result<()> {
        for &(from, to) in copies {
            self.require_unit(from)?;
            self.require_unit(to)?;

            let reason = if from == 0 {
                Some("the source is unit 0, the padding unit")
            } else if to == 0 {
                Some("the destination is unit 0, the padding unit")
            } else if from == to {
                Some("the destination is the source")
            } else {
                None
            };
            if let Some(reason) = reason {
                return Err(Error::Invalid(format!(
                    "cache unit copy {from} -> {to} is refused: {reason}"
                )));
            }
        }
        Ok(())
    }

    /// Copies whole units on `stream`, in stream order: every field a unit holds, whatever
    /// block it is part of, so a copy of an attention, draft or state block is the same copy.
    /// That is the values of every column with their flags, an FP8 cache's scales, the index
    /// keys of a cache that keeps them and the state planes of a cache with a state group. The
    /// copies were checked ([`Self::check_copies`]).
    pub(crate) fn copy_units(&self, copies: &[(u32, u32)], stream: &Stream) -> Result<()> {
        let device = self.device()?;
        let mut batch = CopyBatch::new(SourceOrder::Stream);

        for &(from, to) in copies {
            for column in 0..self.layout.columns {
                for field in 0..2 {
                    batch.push(
                        self.plane_address(field, column, to)?,
                        self.plane_address(field, column, from)?,
                        self.layout.plane_bytes as usize,
                    );
                    batch.push(
                        self.flag_address(field, column, to)?,
                        self.flag_address(field, column, from)?,
                        1,
                    );
                    if device.scales.is_some() {
                        batch.push(
                            self.scale_address(field, column, to)?,
                            self.scale_address(field, column, from)?,
                            4,
                        );
                    }
                }

                if device.index_keys.is_some() {
                    batch.push(
                        self.index_address(column, to)?,
                        self.index_address(column, from)?,
                        self.layout.index_bytes() as usize,
                    );
                }

                for run in 0..device.state.len() {
                    batch.push(
                        self.state_plane_address(run, column, to)?,
                        self.state_plane_address(run, column, from)?,
                        self.layout.plane_bytes as usize,
                    );
                }
            }
        }

        // SAFETY: every range is a cell of a storage this value owns. The buffer pool holds
        // the sources, which nothing writes while they are shared, and gives the destinations
        // to this copy alone until it is reported done, after the batch that runs it ended.
        unsafe { batch.submit(stream) }?;
        Ok(())
    }

    /// The config the value storage `field` was taken with, over `segments`' parts.
    fn segmented_config<'a>(
        device: &'a DevicePlanes,
        segments: &'a Segments,
        field: usize,
    ) -> SegmentedConfig<'a> {
        SegmentedConfig {
            name: VALUES[field],
            device: device.device,
            parts: &segments.parts,
            access: &device.peers,
        }
    }

    /// Gives back the storage of the value planes' segments from `first` on, every column's,
    /// through `owner`, which retained them. The buffer pool gives none of their units to a
    /// block and no device work addresses them any more.
    ///
    /// # Errors
    ///
    /// For planes that are not split, or when `owner` fails.
    pub(crate) fn release_segments(
        &mut self,
        first: u32,
        owner: &dyn ExclusiveStorage,
    ) -> Result<()> {
        let Some(device) = &mut self.device else {
            return Err(Error::State(
                "the cache holds no device: it has no segment to give back",
            ));
        };
        let units = self.units;
        let Some(segments) = &device.segments else {
            return Err(Error::Invalid(format!(
                "the cache planes of {units} units are not split into segments, which is what \
                 gives storage back"
            )));
        };

        let from = segments.released_from;
        if first >= from {
            tracing::info!(
                first,
                released_from = from,
                "no cache segment is left to give back: these were given back already"
            );
            return Ok(());
        }

        let indices = segments.parts_of(first..from);
        for field in 0..device.values.len() {
            owner.release_segments(
                &Self::segmented_config(device, segments, field),
                &device.values[field],
                &indices,
            )?;
        }

        let released = segments.first_unit(from) - segments.first_unit(first);
        if let Some(segments) = &mut device.segments {
            segments.released_from = first;
        }
        tracing::info!(
            first,
            segments = from - first,
            units = released,
            "gave back cache segments"
        );
        Ok(())
    }

    /// Creates the segments given back again at their addresses, the first `limit` of them or
    /// all, through `owner`, and resets their units on `stream`, which is synchronized before
    /// returning: values and flags zero, scales one.
    ///
    /// A wake brings back every asked segment of both value planes or none: a failure part way
    /// is undone here, so the count of held segments still agrees with the mappings and the
    /// next wake asks the server for segments that do not exist.
    ///
    /// # Errors
    ///
    /// For planes that are not split, or when `owner` or the stream fails; an
    /// [`Error::Invariant`] when undoing a failed wake failed too.
    pub(crate) fn restore_segments(
        &mut self,
        owner: &dyn ExclusiveStorage,
        stream: &Stream,
        limit: Option<u32>,
    ) -> Result<()> {
        let (from, count, indices, first_unit, units) = {
            let Some(segments) = self
                .device
                .as_ref()
                .and_then(|device| device.segments.as_ref())
            else {
                return Err(Error::Invalid(format!(
                    "the cache planes of {} units are not split into segments, which is what \
                     gives storage back",
                    self.units
                )));
            };

            let from = segments.released_from;
            let count = limit.map_or(segments.count(), |limit| {
                from.saturating_add(limit).min(segments.count())
            });
            if from == count {
                return Ok(());
            }

            let first_unit = segments.first_unit(from);
            (
                from,
                count,
                segments.parts_of(from..count),
                first_unit,
                segments.first_unit(count) - first_unit,
            )
        };

        let mut restored = 0;
        let mut failure = None;
        {
            let device = self.device()?;
            let segments = device
                .segments
                .as_ref()
                .ok_or(Error::State("the cache's segments are gone"))?;

            for field in 0..device.values.len() {
                let config = Self::segmented_config(device, segments, field);
                if let Err(error) = owner.restore_segments(&config, &device.values[field], &indices)
                {
                    failure = Some(error);
                    break;
                }
                restored += 1;
            }
        }

        if failure.is_none()
            && let Err(error) = self
                .reset_units(first_unit, units, true, stream)
                .and_then(|()| stream.synchronize().map_err(Error::from))
        {
            failure = Some(error);
        }

        if let Some(error) = failure {
            // Put the cache back as asleep as it was: the segments this call created go back
            // through the owner that created them, so its reservation ends with them.
            let device = self.device()?;
            let segments = device
                .segments
                .as_ref()
                .ok_or(Error::State("the cache's segments are gone"))?;

            for field in 0..restored {
                let config = Self::segmented_config(device, segments, field);
                owner
                    .release_segments(&config, &device.values[field], &indices)
                    .map_err(|undo| {
                        Error::Invariant(format!(
                            "a wake failed ({error}), and giving the {restored} storage(s) it \
                             restored back failed too ({undo}): the cache is half woken and its \
                             next wake would ask for segments that exist"
                        ))
                    })?;
            }
            return Err(error);
        }

        if let Some(segments) = self
            .device
            .as_mut()
            .and_then(|device| device.segments.as_mut())
        {
            segments.released_from = count;
        }
        tracing::info!(
            first = from,
            segments = count - from,
            "created the cache segments again"
        );
        Ok(())
    }
}

/// Checks a pool of `units` units of `layout`: the padding unit and one usable unit at least.
fn check(layout: &CacheLayout, units: u32) -> Result<()> {
    if units < 2 {
        return Err(Error::Invalid(
            "a cache unit pool needs the padding unit and one usable unit".into(),
        ));
    }
    if layout.columns == 0 || layout.plane_bytes == 0 {
        return Err(Error::Invalid(
            "a cache unit pool needs columns and planes of bytes".into(),
        ));
    }
    Ok(())
}

/// The runs of the exclusive storage of `config`, cells `[columns, units]` of `cell_bytes`
/// bytes each: one run of every unit per column, named `{name}/{column}`. Empty for storage
/// local to the process.
fn cell_runs(
    owner: &dyn ExclusiveStorage,
    config: &ExclusiveConfig<'_>,
    columns: u32,
    units: u32,
    cell_bytes: u64,
) -> Result<Vec<crate::memory::cache::info::CachePlaneRun>> {
    let Some((key, gpu)) = owner.key(config)? else {
        return Ok(Vec::new());
    };

    Ok((0..columns)
        .map(|column| crate::memory::cache::info::CachePlaneRun {
            field: format!("{}/{column}", config.name),
            key,
            gpu,
            first: 0,
            count: units,
            offset: u64::from(column) * u64::from(units) * cell_bytes,
            bytes: cell_bytes,
        })
        .collect())
}

/// Takes both value planes of `layout` in `parts` segments from `owner`, adding their runs to
/// `runs`. Returns the storages, the segments' state and the segments created for this rank.
fn take_segmented(
    layout: &CacheLayout,
    parts: &[PlaneSegment],
    device: DeviceId,
    peers: &[DeviceId],
    owner: &dyn ExclusiveStorage,
    runs: &mut Vec<crate::memory::cache::info::CachePlaneRun>,
) -> Result<([Arc<Storage>; 2], Segments, Vec<u32>)> {
    let count = parts.len() as u32;
    let columns = layout.columns;
    let plane_bytes = layout.plane_bytes as usize;

    // Part `c · count + k` holds column `c` of segment `k`: its units at a plane's bytes each.
    let segment_parts: Vec<SegmentPart> = (0..columns)
        .flat_map(|column| {
            parts.iter().map(move |part| SegmentPart {
                length: part.segment.units as usize * plane_bytes,
                column,
                segment: part.segment,
                create: part.create,
            })
        })
        .collect();

    let mut storages = Vec::with_capacity(2);
    let mut created = vec![false; parts.len()];
    for name in VALUES {
        let config = SegmentedConfig {
            name,
            device,
            parts: &segment_parts,
            access: peers,
        };
        let (storage, made) = owner.segmented(&config)?;

        let mut first = 0;
        for (index, part) in segment_parts.iter().enumerate() {
            if (index as u32).is_multiple_of(count) {
                first = 0;
            }
            if let Some((key, gpu)) = owner.segment_key(&config, index)?
                && storage.segment_mapped(index)?
            {
                runs.push(crate::memory::cache::info::CachePlaneRun {
                    field: format!("{name}/{}", part.column),
                    key,
                    gpu,
                    first,
                    count: part.segment.units,
                    offset: 0,
                    bytes: layout.plane_bytes,
                });
            }
            first += part.segment.units;

            created[index % parts.len()] |= made.get(index).copied().unwrap_or(false);
        }
        storages.push(storage);
    }
    let [key, value]: [Arc<Storage>; 2] = storages
        .try_into()
        .map_err(|_| Error::Invariant("a cache holds a key plane and a value plane".into()))?;

    // Segments given back, which no rank of the region holds, are the same ones of every
    // column of both planes, from one segment on.
    let mapped = |segment: u32, column: u32| -> Result<(bool, bool)> {
        let index = (column * count + segment) as usize;
        Ok((key.segment_mapped(index)?, value.segment_mapped(index)?))
    };
    let mut released_from = count;
    for segment in 0..count {
        let mut all_gone = true;
        for column in 0..columns {
            let (key, value) = mapped(segment, column)?;
            all_gone &= !key && !value;
        }
        if all_gone {
            released_from = segment;
            break;
        }
    }
    for column in 0..columns {
        for segment in 0..count {
            let (key, value) = mapped(segment, column)?;
            let gone = !key || !value;
            if gone != (segment >= released_from) {
                return Err(Error::Invalid(format!(
                    "the cache planes' segment {segment} of column {column} is {} while the \
                     planes are given back from segment {released_from}",
                    if gone { "unmapped" } else { "mapped" }
                )));
            }
        }
    }

    let segments = Segments {
        segments: parts.iter().map(|part| part.segment).collect(),
        released_from,
        columns,
        parts: segment_parts,
    };
    let created = (0..count)
        .filter(|segment| created[*segment as usize])
        .collect();

    Ok(([key, value], segments, created))
}

/// Fills `rows` rows of `count` elements of `width` bytes with `value`, `pitch` bytes apart,
/// from `address`, on `stream`. An element is one byte or four.
///
/// # Safety
///
/// Every row lies in mapped device storage that stays valid until the fill completes.
unsafe fn fill_rows(
    stream: &Stream,
    address: u64,
    pitch: usize,
    value: u32,
    width: usize,
    count: usize,
    rows: usize,
) -> Result<()> {
    // SAFETY: forwarded from the caller.
    unsafe {
        match width {
            1 => memset_2d_d8(address, pitch, value as u8, count, rows, stream)?,
            _ => memset_2d_d32(address, pitch, value, count, rows, stream)?,
        }
    }
    Ok(())
}

/// Fills `count` elements of `width` bytes with `value` from `address` on `stream`.
///
/// # Safety
///
/// As [`fill_rows`].
unsafe fn fill(
    stream: &Stream,
    address: u64,
    value: u32,
    width: usize,
    count: usize,
) -> Result<()> {
    // SAFETY: forwarded from the caller.
    unsafe { fill_rows(stream, address, count * width, value, width, count, 1) }
}

#[cfg(test)]
mod tests {
    use nn::family::IndexKeys;
    use proto::request::PrefixCacheDtype;

    use super::*;

    /// Planes of two layers of 16-token pages of one 8-wide head, over eight units, with no
    /// device.
    fn planes(dtype: PrefixCacheDtype) -> Planes {
        let layout = CacheLayout::dense(16, 2, 1, 8, dtype)
            .validated()
            .expect("a dense layout");
        Planes::meta(Arc::new(layout), 8).expect("planes of eight units")
    }

    #[test]
    fn runs_of_units_are_consecutive_and_each_unit_counts_once() {
        assert_eq!(unit_runs(&[5, 3, 4, 9, 4]), vec![(3, 3), (9, 1)]);
        assert!(unit_runs(&[]).is_empty());
    }

    /// A layer's planes are its column's units, and a dense cache states no scale.
    #[test]
    fn a_layer_is_one_column_of_every_unit() -> Result<()> {
        let dense = planes(PrefixCacheDtype::BFloat16);
        let group = dense.layout.groups[0].clone();

        let layer = dense.layer(&group, 1)?;
        assert_eq!(layer.key.dims(), [8, 16, 1, 8]);
        assert_eq!(layer.value.dtype(), DType::BF16);
        assert_eq!(layer.key_initialized.dims(), [8, 1, 1, 1]);
        assert_eq!(layer.value_initialized.dtype(), DType::U8);
        assert!(layer.key_scale.is_none() && layer.value_scale.is_none());
        assert!(layer.index_keys.is_none());
        assert!(dense.layer(&group, 2).is_err());

        let fp8 = planes(PrefixCacheDtype::Float8E4m3Fn);
        let layer = fp8.layer(&fp8.layout.groups[0].clone(), 0)?;
        assert_eq!(layer.key.dtype(), DType::F8E4M3);
        let scale = layer.key_scale.expect("an FP8 cache states its scales");
        assert_eq!(
            (scale.dims(), scale.dtype()),
            (&[8, 1, 1, 1][..], DType::F32)
        );
        Ok(())
    }

    /// A layer of a cache that keeps index keys lends them on its pages: one key of the stated
    /// width per `ratio` tokens of each unit of its column, in the layers' compute type
    /// whatever the keys and values are stored in.
    #[test]
    fn a_layer_lends_the_index_keys_of_its_pages() -> Result<()> {
        let keys = IndexKeys { width: 6, ratio: 4 };

        for dtype in [PrefixCacheDtype::BFloat16, PrefixCacheDtype::Float8E4m3Fn] {
            let layout = CacheLayout::dense(16, 2, 1, 8, dtype).with_index_keys(keys)?;
            let group = layout.groups[0].clone();
            let planes = Planes::meta(Arc::new(layout), 8)?;

            let layer = planes.layer(&group, 1)?;
            let index_keys = layer.index_keys.expect("the layer keeps index keys");
            assert_eq!(
                (index_keys.dims(), index_keys.dtype()),
                (&[8, 4, 6][..], DType::BF16)
            );
            assert_eq!(layer.key.dims(), [8, 16, 1, 8]);
        }
        Ok(())
    }

    /// A draft layer's index keys lie over the columns its pages do: one page per cell.
    #[test]
    fn a_draft_layer_lends_the_index_keys_of_its_span() -> Result<()> {
        let keys = IndexKeys { width: 6, ratio: 4 };
        let layout = CacheLayout::dense(16, 4, 1, 8, PrefixCacheDtype::BFloat16)
            .with_index_keys(keys)?
            .with_draft(1, 1, 8)?;
        let draft = layout.groups[1].clone();
        let span = layout.draft_span().expect("the layout holds a draft group");
        assert_eq!(span, 4);
        let planes = Planes::meta(Arc::new(layout), 8)?;

        let layer = planes.draft_layer(&draft, 0, span)?;
        assert_eq!(layer.key.dims(), [32, 16, 1, 8]);
        assert_eq!(
            layer
                .index_keys
                .expect("the drafter keeps index keys")
                .dims(),
            [32, 4, 6]
        );
        Ok(())
    }

    /// A refused copy names the pair and the reason.
    #[test]
    fn a_refused_unit_copy_names_the_pair_and_the_reason() {
        let planes = planes(PrefixCacheDtype::BFloat16);
        let refused = |from: u32, to: u32| {
            planes
                .check_copies(&[(from, to)])
                .expect_err("this copy is refused")
                .to_string()
        };

        let padding = refused(1, 0);
        assert!(padding.contains("1 -> 0"), "{padding}");
        assert!(padding.contains("padding unit"), "{padding}");
        assert!(refused(2, 2).contains("the destination is the source"));
        assert!(refused(1, 8).contains("outside the pool"));
        assert!(planes.check_copies(&[(1, 2)]).is_ok());
    }

    /// Planes that hold no device map nothing, hold no segment and refuse device work.
    #[test]
    fn planes_with_no_device_have_no_storage() {
        let mut planes = planes(PrefixCacheDtype::BFloat16);

        assert_eq!(planes.mapped_bytes(), 0);
        assert!(planes.segments().is_none());
        assert_eq!(planes.held_segments(), 0);
        assert!(planes.runs().is_empty());
        assert!(planes.plane_address(0, 0, 1).is_err());
        assert!(planes.index_address(0, 1).is_err());
        assert!(
            planes
                .release_segments(0, &crate::memory::exclusive::LocalStorage)
                .is_err()
        );
    }

    /// A pool needs the padding unit and one unit a page can take.
    #[test]
    fn a_pool_of_one_unit_is_refused() {
        let layout = CacheLayout::dense(16, 2, 1, 8, PrefixCacheDtype::BFloat16);
        assert!(Planes::meta(Arc::new(layout), 1).is_err());
    }
}
