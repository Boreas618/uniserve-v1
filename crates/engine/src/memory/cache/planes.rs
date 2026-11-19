//! The cache's planes: the device storage of a pool of units and its unit-level resets and copies.
//!
//! Each field of a pool ([`super::layout::PoolLayout`]) owns three storages, column-major, a cell
//! being one column (an owner) of one unit:
//!
//! | Storage | Element | Shape |
//! | --- | --- | --- |
//! | `prefix_cache.{field}.values` | bytes | column `c`: `[units, plane bytes of c]`, columns back to back |
//! | `prefix_cache.{field}.initialized` | U8 | `[columns, units]` |
//! | `prefix_cache.{field}.scale` (a field in the deployment's FP8 only) | F32 | `[columns, units]` |
//!
//! The names key the KV pool's storages at the Storage Server, where every epoch that names the
//! cache's region and a restarted engine find them, so they stay as they are whatever this module
//! is called; a model of keys and values names its fields `key` and `value`, and an indexer's keys
//! are `index_key`, the names a handoff between replicas reads a rank's fields by
//! (`storage::protocol::planes`). Every storage of the KV pool is the memory of the region: the
//! segments of the value storages are keyed by the segments the region's segment table names, and
//! the other storages by the epoch of its first segment, the epoch whose ranks created the
//! region's memory. Two epochs of one layout therefore address the same cells, and the worker that
//! is given a unit resets its flags and scales before a batch writes it ([`Planes::reset_units`]).
//! A sliding window's pool is memory of the rank's own process ([`LocalStorage`]): nothing in it
//! outlives a request, so no other epoch maps it.
//!
//! A field group's page is one unit and each owner one column, so an owner's entries are the
//! contiguous `[units, page / ratio, heads, width]` of its column and a page id is a unit id. A
//! draft layer lies over `span` columns of the KV pool's keys and values, so its pages are the
//! `span · units` planes of those columns.
//!
//! On a device whose KV pool splits ([`super::segments`]), each value storage is segmented alike:
//! the last segments are given back ([`Planes::release_segments`]) and created again at their
//! addresses ([`Planes::restore_segments`]). The flags and the scales are not segmented: they are
//! one storage each that stays mapped, a byte or four of a plane. The state region's units are
//! split by the same functions ([`take_split`], [`release_split`], [`restore_split`]) over
//! segments of their own ([`super::states`]).

use std::sync::Arc;

use nn::KvPlane;
use platform::{CopyBatch, DeviceId, SourceOrder, Stream, memset_2d_d8, memset_2d_d32};
use tensor::{DType, Tensor};

use super::layout::{FieldLayout, GroupLayout, PoolLayout};
use super::segments::{self, PlaneSegment};
use crate::memory::allocation::Storage;
use crate::memory::exclusive::{
    ExclusiveConfig, ExclusiveStorage, LocalStorage, SegmentPart, SegmentedConfig,
};
use crate::memory::lend;
use crate::worker::{Error, Result};

/// The neutral FP8 scale, `1.0f32`, as the bits a 32-bit fill writes: what every scale of a new
/// unit of the planes and of a scratch page set holds.
pub(super) const UNIT_SCALE: u32 = 1.0_f32.to_bits();

/// The names of the storages of field `name`: its entries, its initialization flags and its page
/// scales.
pub(crate) fn storage_names(name: &str) -> [String; 3] {
    ["values", "initialized", "scale"].map(|part| format!("prefix_cache.{name}.{part}"))
}

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

/// One storage of a split set: its name and the bytes one unit takes in each of its columns.
pub(super) type SplitStorage = (String, Vec<u64>);

/// The segments of a split storage set (a device cache's value storages, or a state region's
/// units) and which of them are mapped.
pub(super) struct Segments {
    /// The segments in unit order.
    segments: Vec<crate::memory::cache::info::CacheSegment>,
    /// The first segment given back; the segment count while none is.
    released_from: u32,
    /// Each split storage's name and parts, column-major: the part of segment `k` that column
    /// `c` holds is `c · segments.len() + k`, in the order the storages are kept.
    storages: Vec<(String, Vec<SegmentPart>)>,
}

impl Segments {
    /// The segments in unit order.
    pub(super) fn segments(&self) -> &[crate::memory::cache::info::CacheSegment] {
        &self.segments
    }

    /// The segments held, `[0, released_from)`.
    pub(super) fn held(&self) -> u32 {
        self.released_from
    }

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

    /// The units of `segments`, ascending segment indices, as runs `(first unit, units)`: the
    /// segments that lie back to back join one run, so a cache of many segments resets what it
    /// created with a fill per run rather than per segment.
    fn unit_runs(&self, segments: &[u32]) -> Vec<(u32, u32)> {
        let mut runs: Vec<(u32, u32)> = Vec::new();
        let (mut next, mut first) = (0, 0);

        for &index in segments {
            // The first unit of segment `index`, summed on from the last segment visited.
            for segment in &self.segments[next as usize..index as usize] {
                first += segment.units;
            }
            let units = self.segments[index as usize].units;
            next = index + 1;

            match runs.last_mut() {
                Some((start, count)) if *start + *count == first => *count += units,
                _ => runs.push((first, units)),
            }
            first += units;
        }
        runs
    }

    /// The bytes the parts of the segments given back take over every split storage.
    pub(super) fn released_bytes(&self) -> u64 {
        (0..self.storages.len())
            .flat_map(|storage| {
                let parts = &self.storages[storage].1;
                self.parts_of(storage, self.released_from..self.count())
                    .into_iter()
                    .map(move |part| parts[part].length as u64)
            })
            .sum()
    }

    /// The parts of segments `range` of split storage `storage`, every column's.
    fn parts_of(&self, storage: usize, range: std::ops::Range<u32>) -> Vec<usize> {
        let count = self.count();
        let columns = self.storages[storage].1.len() as u32 / count.max(1);

        (0..columns)
            .flat_map(|column| {
                range
                    .clone()
                    .map(move |segment| (column * count + segment) as usize)
            })
            .collect()
    }
}

/// The storages of one field of a pool on a device.
struct FieldStorages {
    /// The entries of every column, column-major.
    values: Arc<Storage>,
    /// One U8 flag per cell.
    initialized: Arc<Storage>,
    /// One F32 scale per cell, for a scaled field.
    scale: Option<Arc<Storage>>,
}

/// The storages of a pool on a device.
struct DevicePlanes {
    device: DeviceId,
    /// Other local devices that address the planes: readers there copy pages directly.
    peers: Vec<DeviceId>,
    /// One per field of the pool, in its order.
    fields: Vec<FieldStorages>,
    /// The segments of the value storages, when the pool splits into them.
    segments: Option<Segments>,
    /// Where the Storage Server retains every unit's bytes, run by run.
    runs: Vec<crate::memory::cache::info::CachePlaneRun>,
}

/// A pool's storage: on one device, or none for a cache that holds no device, whose tensors are
/// meta tensors.
pub(crate) struct Planes {
    pool: PoolLayout,
    units: u32,
    device: Option<DevicePlanes>,
}

impl Planes {
    /// Planes of `units` units of `pool` that hold no storage: every tensor is a meta tensor.
    pub(crate) fn meta(pool: PoolLayout, units: u32) -> Result<Self> {
        check(&pool, units)?;

        Ok(Self {
            pool,
            units,
            device: None,
        })
    }

    /// Takes the planes of `units` units of `pool` (the padding unit included) on `device` from
    /// `owner`, addressed by `device` and each of `peers`, the value storages in `parts`
    /// segments when the pool splits, and resets what was created for this rank on `stream`,
    /// which is synchronized before returning: values and flags zero, scales one.
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
    /// When the segments do not hold `units` units, the segments of one storage are not given
    /// back from one segment on in every column, `owner` refuses a storage, or a reset fails.
    pub(crate) fn open(
        pool: PoolLayout,
        units: u32,
        parts: Option<&[PlaneSegment]>,
        (device, peers): (DeviceId, &[DeviceId]),
        owner: &dyn ExclusiveStorage,
        stream: &Stream,
    ) -> Result<Self> {
        check(&pool, units)?;
        if parts.is_some_and(|parts| segments::units(parts) != units) {
            return Err(Error::Invalid(
                "a cache's segments do not hold its units".into(),
            ));
        }

        let peers = peers.to_vec();
        let block = pool.block_size;
        let names: Vec<[String; 3]> = pool
            .fields
            .iter()
            .map(|field| storage_names(field.name))
            .collect();
        // The bytes of every column's plane of each field.
        let columns: Vec<Vec<u64>> = pool
            .fields
            .iter()
            .map(|field| {
                (0..field.owners.len())
                    .map(|column| field.plane_bytes(column, block))
                    .collect()
            })
            .collect();

        // The epoch whose ranks created the region's memory: that of its first segment, which
        // holds unit 0. A segment given back and created again keeps its identity, so the epoch
        // stays the same for the region's life.
        let epoch = parts
            .and_then(|parts| parts.first())
            .map(|part| part.segment.epoch);
        let held = (device, peers.as_slice(), epoch);

        // The exclusive storages, field by field in the order they are taken apart below: the
        // values of a pool that does not split, the flags, and the scales of a scaled field. A
        // split pool takes its values in segments.
        let units_bytes = u64::from(units);
        let mut configs = Vec::new();
        let mut cells = Vec::new();
        for ((field, names), columns) in pool.fields.iter().zip(&names).zip(&columns) {
            let cell_count = field.owners.len() * units as usize;
            if parts.is_none() {
                let bytes: u64 = columns.iter().sum::<u64>() * units_bytes;
                configs.push(exclusive(&names[0], bytes as usize, held));
                cells.push(columns.clone());
            }
            configs.push(exclusive(&names[1], cell_count, held));
            cells.push(vec![1; field.owners.len()]);
            if field.scaled {
                configs.push(exclusive(&names[2], cell_count * 4, held));
                cells.push(vec![4; field.owners.len()]);
            }
        }

        let mut runs = Vec::new();
        for (config, cells) in configs.iter().zip(&cells) {
            runs.extend(cell_runs(owner, config, cells, units)?);
        }
        let mut taken = owner.exclusive(&configs)?.into_iter();
        let mut next = || -> Result<(Arc<Storage>, bool)> {
            taken
                .next()
                .ok_or_else(|| Error::Invariant("an exclusive storage is missing".into()))
        };

        // The values of a split pool, in segments, one split storage per field.
        let mut segmented = None;
        let mut created_segments: Vec<u32> = Vec::new();
        let mut split_values = Vec::new();
        let mut values_created = false;
        if let Some(parts) = parts {
            let storages: Vec<SplitStorage> = names
                .iter()
                .zip(&columns)
                .map(|(names, columns)| (names[0].clone(), columns.clone()))
                .collect();
            let (storages, segments, created) =
                take_split(&storages, parts, (device, &peers), owner, &mut runs)?;
            values_created = created.len() == parts.len();
            if !values_created {
                created_segments = created;
            }
            segmented = Some(segments);
            split_values = storages;
        }

        let mut split_values = split_values.into_iter();
        let mut fields = Vec::with_capacity(pool.fields.len());
        let mut created = Vec::with_capacity(pool.fields.len());
        for field in &pool.fields {
            let (values, values_made) = match parts {
                None => next()?,
                Some(_) => (
                    split_values.next().ok_or_else(|| {
                        Error::Invariant("a split value storage is missing".into())
                    })?,
                    values_created,
                ),
            };
            let (initialized, flags_made) = next()?;
            let (scale, scale_made) = match field.scaled {
                true => {
                    let (scale, made) = next()?;
                    (Some(scale), made)
                }
                false => (None, false),
            };
            fields.push(FieldStorages {
                values,
                initialized,
                scale,
            });
            created.push((values_made, flags_made, scale_made));
        }

        let planes = Self {
            pool,
            units,
            device: Some(DevicePlanes {
                device,
                peers,
                fields,
                segments: segmented,
                runs,
            }),
        };
        let held = planes.device()?;

        for ((storages, &(values, flags, scales)), field) in
            held.fields.iter().zip(&created).zip(&planes.pool.fields)
        {
            let cells = field.owners.len() * units as usize;
            // SAFETY: each fill spans exactly one storage this value owns, whose segments are
            // all mapped when its values were created as a whole; the stream is drained below.
            unsafe {
                if values {
                    let bytes = field.unit_bytes(block) as usize * units as usize;
                    fill(stream, storages.values.address(), 0, 1, bytes)?;
                }
                if flags {
                    fill(stream, storages.initialized.address(), 0, 1, cells)?;
                }
                if let Some(scale) = &storages.scale
                    && scales
                {
                    fill(stream, scale.address(), UNIT_SCALE, 4, cells)?;
                }
            }
        }
        if let Some(segments) = &held.segments {
            for (first, count) in segments.unit_runs(&created_segments) {
                planes.reset_units(first, count, true, stream)?;
            }
        }
        stream.synchronize()?;

        Ok(planes)
    }

    /// Takes the planes of `units` units of a sliding window's `pool` on `device` from the
    /// process's own memory ([`LocalStorage`]), addressed by `device` and each of `peers`, and
    /// resets them on `stream`, which is synchronized before returning: a window's pages are a
    /// running request's alone, so no other process or epoch maps them.
    ///
    /// # Errors
    ///
    /// As [`Self::open`].
    pub(crate) fn open_local(
        pool: PoolLayout,
        units: u32,
        (device, peers): (DeviceId, &[DeviceId]),
        stream: &Stream,
    ) -> Result<Self> {
        Self::open(pool, units, None, (device, peers), &LocalStorage, stream)
    }

    /// The units of the pool, the padding unit included.
    pub(crate) fn units(&self) -> u32 {
        self.units
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

    /// The segments the value storages are made of, in unit order; `None` for planes that are
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

    /// The device bytes one unit of the split storages holds over every column of every field
    /// ([`PoolLayout::split_bytes`]): what a segment gives back per unit, which the flags and
    /// scales are no part of.
    pub(crate) fn unit_bytes(&self) -> u64 {
        self.pool.split_bytes()
    }

    /// The bytes the planes map on the device: the value storages' mapped segments, the flags
    /// and the scales. A sleeping rank's segments are not in it.
    pub(crate) fn mapped_bytes(&self) -> u64 {
        let Some(device) = &self.device else {
            return 0;
        };
        let bytes = |storage: &Arc<Storage>| storage.bytes() as u64;

        // The storages keep their length while their segments are unmapped.
        let released = device.segments.as_ref().map_or(0, Segments::released_bytes);
        let held: u64 = device
            .fields
            .iter()
            .map(|field| {
                bytes(&field.values)
                    + bytes(&field.initialized)
                    + field.scale.as_ref().map_or(0, bytes)
            })
            .sum();
        held.saturating_sub(released)
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

    /// The field at `field` of the pool, or the refusal of one the pool does not hold.
    fn field(&self, field: usize) -> Result<&FieldLayout> {
        self.pool.fields.get(field).ok_or_else(|| {
            Error::Invalid(format!(
                "the pool holds {} fields, not field {field}",
                self.pool.fields.len()
            ))
        })
    }

    /// The tensor of `shape` elements of `dtype` whose first element lies `offset` bytes into
    /// `storage` of field `field` of the device side, or the meta tensor of that shape.
    fn tensor(
        &self,
        storage: impl FnOnce(&FieldStorages) -> Option<&Arc<Storage>>,
        (field, offset): (usize, u64),
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        match &self.device {
            Some(device) => {
                let held =
                    device.fields.get(field).and_then(storage).ok_or_else(|| {
                        Error::Invariant(format!("field {field} holds no storage"))
                    })?;
                // SAFETY: the planes own the storage for as long as they live, and the holder of
                // a cache keeps it for as long as tensors over it and work issued on them exist;
                // a window over segments given back is addressed by no unit the engine
                // allocates.
                unsafe { held.external(offset as usize, shape, dtype) }
            }
            // SAFETY: a meta tensor has no data and its address is never read.
            None => unsafe { lend(0, shape, dtype, None) },
        }
    }

    /// The planes of `group`, a field group whose pages are units of this pool: every owner of
    /// every field, fields in order and owners ascending, each its column's entries as
    /// `[units, block_size / ratio, heads, width]` with the flags and scales of the same cells
    /// as `[units, 1, 1, 1]`. A page id of the group is a unit id.
    pub(crate) fn planes(&self, group: &GroupLayout) -> Result<Vec<KvPlane>> {
        let mut planes = Vec::new();
        for (index, field) in self.pool.fields.iter().enumerate() {
            for (column, owner) in field.owners.iter().enumerate() {
                planes.push(self.columns(
                    (index, field),
                    (column, 1),
                    owner.layer,
                    group.block_size,
                )?);
            }
        }
        Ok(planes)
    }

    /// The planes of the draft group `group`'s layers, lying over `span` columns each of every
    /// field of the KV pool: layer `k`'s plane of a field is its columns `k · span ..` as pages of
    /// `block_size / span` tokens, `[span · units, block_size / span, heads, width]`, page
    /// `c · units + u` holding the `c`-th run of the draft page in unit `u`, with the flags and
    /// scales of the same cells. Planes are in the drafter's layer order, fields in order within
    /// a layer.
    pub(crate) fn draft_planes(&self, group: &GroupLayout, span: u32) -> Result<Vec<KvPlane>> {
        let mut planes = Vec::new();
        for layer in 0..group.layers.len() as u32 {
            for (index, field) in self.pool.fields.iter().enumerate() {
                let first = (layer * span) as usize;
                if first + span as usize > field.owners.len() {
                    return Err(Error::Invalid(format!(
                        "the draft group's layer {layer} lies over columns {first}.. of the field \
                         {}, which holds {}",
                        field.name,
                        field.owners.len()
                    )));
                }
                planes.push(self.columns(
                    (index, field),
                    (first, span as usize),
                    layer,
                    group.block_size / span,
                )?);
            }
        }
        Ok(planes)
    }

    /// The plane of `columns` columns of `field` from `first`, as pages of `tokens` tokens, lent
    /// as `layer`'s: the columns' cells lie contiguously in each storage, so columns of equal
    /// planes are one tensor of their pages.
    fn columns(
        &self,
        (index, field): (usize, &FieldLayout),
        (first, columns): (usize, usize),
        layer: u32,
        tokens: u32,
    ) -> Result<KvPlane> {
        let owner = field.owners[first];
        let block = self.pool.block_size;
        let entries = tokens / owner.ratio;
        let bytes = u64::from(entries) * field.entry_bytes();
        if (first..first + columns).any(|column| field.plane_bytes(column, block) != bytes) {
            return Err(Error::Invariant(format!(
                "pages of {tokens} tokens of the field {} do not fill its columns {first}..{}",
                field.name,
                first + columns
            )));
        }

        let pages = columns * self.units as usize;
        let cell = first as u64 * u64::from(self.units);
        let offset = field.column_offset(first, self.units, block);
        let [heads, width] = field.dims;
        let values = [pages, entries as usize, heads as usize, width as usize];
        let cells = [pages, 1, 1, 1];

        Ok(KvPlane {
            field: field.name,
            layer,
            readers: if columns == 1 {
                field
                    .readers
                    .iter()
                    .filter(|reader| reader.owner == owner.layer)
                    .map(|reader| reader.layer)
                    .collect()
            } else {
                Vec::new()
            },
            ratio: owner.ratio,
            entries: self.tensor(
                |storages| Some(&storages.values),
                (index, offset),
                &values,
                field.dtype,
            )?,
            initialized: self.tensor(
                |storages| Some(&storages.initialized),
                (index, cell),
                &cells,
                DType::U8,
            )?,
            scale: field
                .scaled
                .then(|| {
                    self.tensor(
                        |storages| storages.scale.as_ref(),
                        (index, cell * 4),
                        &cells,
                        DType::F32,
                    )
                })
                .transpose()?,
        })
    }

    /// The address of field `field`'s plane in `column` of `unit`.
    pub(crate) fn plane_address(&self, field: usize, column: u32, unit: u32) -> Result<u64> {
        self.require_unit(unit)?;
        let laid = self.field(field)?;
        let block = self.pool.block_size;
        let offset = laid.column_offset(column as usize, self.units, block)
            + u64::from(unit) * laid.plane_bytes(column as usize, block);
        Ok(self.device()?.fields[field].values.address() + offset)
    }

    /// The address of field `field`'s FP8 scale in `column` of `unit`.
    pub(crate) fn scale_address(&self, field: usize, column: u32, unit: u32) -> Result<u64> {
        self.require_unit(unit)?;
        let scale = self.device()?.fields[field].scale.as_ref().ok_or_else(|| {
            Error::Invalid(format!(
                "field {field} of the cache's planes carries no scale"
            ))
        })?;
        Ok(scale.address() + self.cell(column, unit) * 4)
    }

    /// The address of field `field`'s initialization flag in `column` of `unit`.
    pub(crate) fn flag_address(&self, field: usize, column: u32, unit: u32) -> Result<u64> {
        self.require_unit(unit)?;
        Ok(self.device()?.fields[field].initialized.address() + self.cell(column, unit))
    }

    /// The index of the cell of `column` and `unit` in a field's flags and scales.
    fn cell(&self, column: u32, unit: u32) -> u64 {
        u64::from(column) * u64::from(self.units) + u64::from(unit)
    }

    /// Resets units `[first, first + count)` of every column of every field with fills issued
    /// on `stream`: flags zero, scales one, and the values zero when `values`.
    ///
    /// The values are filled with one fill per column: a column's units are consecutive and
    /// mapped, while the columns of a split cache lie apart by the segments given back from each
    /// column's tail, which a fill's range may not span although no row touches them. The values
    /// are reset only where units are created, so their fills are rare; the flags and scales,
    /// reset with every allocation, are storages that never split and take one strided fill
    /// each.
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
        let block = self.pool.block_size;

        for (storages, field) in device.fields.iter().zip(&self.pool.fields) {
            let columns = field.owners.len();
            if values {
                for column in 0..columns {
                    let plane = field.plane_bytes(column, block);
                    let base = storages.values.address()
                        + field.column_offset(column, self.units, block)
                        + u64::from(first) * plane;
                    // SAFETY: the range lies inside a storage this value owns: units
                    // `[first, first + count)` of one column. The caller has those units mapped
                    // and orders the fills before their readers.
                    unsafe { fill(stream, base, 0, 1, (u64::from(count) * plane) as usize) }?;
                }
            }

            // SAFETY: every row lies inside a storage this value owns: units
            // `[first, first + count)` of each column, a column's units apart. The caller has
            // those units mapped and orders the fills before their readers.
            unsafe {
                let flags = storages.initialized.address() + u64::from(first);
                fill_rows(stream, flags, units, 0, 1, count as usize, columns)?;

                if let Some(scale) = &storages.scale {
                    let scales = scale.address() + u64::from(first) * 4;
                    fill_rows(
                        stream,
                        scales,
                        units * 4,
                        UNIT_SCALE,
                        4,
                        count as usize,
                        columns,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Raises every field's initialization flags of `units` in every column on `stream`: the
    /// units' values and scales are in place and a reader may use them.
    pub(crate) fn mark_initialized(&self, units: &[u32], stream: &Stream) -> Result<()> {
        let device = self.device()?;

        for (first, count) in unit_runs(units) {
            self.require_unit(first + count - 1)?;

            for (storages, field) in device.fields.iter().zip(&self.pool.fields) {
                // SAFETY: the rows lie inside the flag storage this value owns, and the caller
                // orders the fill behind the copies that filled the units.
                unsafe {
                    fill_rows(
                        stream,
                        storages.initialized.address() + u64::from(first),
                        self.units as usize,
                        1,
                        1,
                        count as usize,
                        field.owners.len(),
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
    /// block it is part of, so a copy of an attention or a draft block is the same copy. That is
    /// the planes of every column of every field with their flags and the scales of a scaled
    /// field. The copies were checked ([`Self::check_copies`]).
    pub(crate) fn copy_units(&self, copies: &[(u32, u32)], stream: &Stream) -> Result<()> {
        let device = self.device()?;
        let block = self.pool.block_size;
        let mut batch = CopyBatch::new(SourceOrder::Stream);

        for &(from, to) in copies {
            for (index, (storages, field)) in
                device.fields.iter().zip(&self.pool.fields).enumerate()
            {
                for column in 0..field.owners.len() as u32 {
                    batch.push(
                        self.plane_address(index, column, to)?,
                        self.plane_address(index, column, from)?,
                        field.plane_bytes(column as usize, block) as usize,
                    );
                    batch.push(
                        self.flag_address(index, column, to)?,
                        self.flag_address(index, column, from)?,
                        1,
                    );
                    if storages.scale.is_some() {
                        batch.push(
                            self.scale_address(index, column, to)?,
                            self.scale_address(index, column, from)?,
                            4,
                        );
                    }
                }
            }
        }

        // SAFETY: every range is a cell of a storage this value owns. The buffer pool holds
        // the sources, which nothing writes while they are shared, and gives the destinations
        // to this copy alone until it is reported done, after the batch that runs it ended.
        unsafe { batch.submit(stream) }?;
        Ok(())
    }

    /// Gives back the storage of the segments of the value storages from `first` on, every
    /// column's, through `owner`, which retained them. The buffer pool gives none of their units
    /// to a block and no device work addresses them any more.
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
        let storages: Vec<Arc<Storage>> = device
            .fields
            .iter()
            .map(|field| Arc::clone(&field.values))
            .collect();
        let Some(segments) = &mut device.segments else {
            return Err(Error::Invalid(format!(
                "the cache planes of {} units are not split into segments, which is what \
                 gives storage back",
                self.units
            )));
        };

        release_split(
            (device.device, &device.peers),
            segments,
            &storages,
            first,
            owner,
        )
    }

    /// Creates the segments given back again at their addresses, the first `limit` of them or
    /// all, through `owner`, and resets their units on `stream`, which is synchronized before
    /// returning: values and flags zero, scales one ([`restore_split`]).
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
        let units = self.units;
        let Some(device) = self.device.as_ref() else {
            return Err(Error::State("the cache holds no device: it has no segment"));
        };
        let storages: Vec<Arc<Storage>> = device
            .fields
            .iter()
            .map(|field| Arc::clone(&field.values))
            .collect();
        let (device_id, peers) = (device.device, device.peers.clone());

        // The segments are taken out of the planes for the call, so the reset reads the planes
        // while the segments change, and are put back whatever the call returns.
        let Some(mut segments) = self
            .device
            .as_mut()
            .and_then(|device| device.segments.take())
        else {
            return Err(Error::Invalid(format!(
                "the cache planes of {units} units are not split into segments, which is what \
                 gives storage back"
            )));
        };
        let restored = restore_split(
            (device_id, &peers),
            &mut segments,
            &storages,
            (owner, limit),
            |first, count| {
                self.reset_units(first, count, true, stream)?;
                stream.synchronize().map_err(Error::from)
            },
        );
        if let Some(device) = self.device.as_mut() {
            device.segments = Some(segments);
        }
        restored
    }
}

/// The config the split storage `storage` of `segments` was taken with, on `device`, which
/// `peers` address.
fn segmented_config<'a>(
    (device, peers): (DeviceId, &'a [DeviceId]),
    segments: &'a Segments,
    storage: usize,
) -> SegmentedConfig<'a> {
    let (name, parts) = &segments.storages[storage];
    SegmentedConfig {
        name,
        device,
        parts,
        access: peers,
    }
}

/// The exclusive storage `name` of `bytes` bytes on `device`, addressed by `access`, keyed by
/// `epoch` ([`ExclusiveConfig::epoch`]).
fn exclusive<'a>(
    name: &'a str,
    bytes: usize,
    (device, access, epoch): (DeviceId, &'a [DeviceId], Option<u64>),
) -> ExclusiveConfig<'a> {
    ExclusiveConfig {
        name,
        device,
        bytes,
        access,
        epoch,
    }
}

/// Gives back the storage of the segments of a split set from `first` on, every column's part of
/// each of `storages` (in the order `segments` keeps them), through `owner`, which retained them.
/// The buffer pool gives none of their units to a block and no device work addresses them any
/// more. Segments given back already are not given back twice.
///
/// # Errors
///
/// When `owner` fails.
pub(super) fn release_split(
    device: (DeviceId, &[DeviceId]),
    segments: &mut Segments,
    storages: &[Arc<Storage>],
    first: u32,
    owner: &dyn ExclusiveStorage,
) -> Result<()> {
    let from = segments.released_from;
    if first >= from {
        tracing::info!(
            first,
            released_from = from,
            "no segment is left to give back: these were given back already"
        );
        return Ok(());
    }

    for (index, storage) in storages.iter().enumerate() {
        let indices = segments.parts_of(index, first..from);
        owner.release_segments(
            &segmented_config(device, segments, index),
            storage,
            &indices,
        )?;
    }

    let released = segments.first_unit(from) - segments.first_unit(first);
    segments.released_from = first;
    tracing::info!(
        names = ?segments.storages.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        first,
        segments = from - first,
        units = released,
        "gave back segments"
    );
    Ok(())
}

/// Creates the segments of a split set given back again at their addresses, the first `limit`
/// of them or all, through `owner`, every column's part of each of `storages`, then has `reset`
/// prepare their units (`(first unit, units)`, the stream drained when it returns).
///
/// A wake brings back every asked segment of every split storage or none: a failure part way is
/// undone here, so the count of held segments still agrees with the mappings and the next wake
/// asks the server for segments that do not exist.
///
/// # Errors
///
/// When `owner` or `reset` fails; an [`Error::Invariant`] when undoing a failed wake failed too.
pub(super) fn restore_split(
    device: (DeviceId, &[DeviceId]),
    segments: &mut Segments,
    storages: &[Arc<Storage>],
    (owner, limit): (&dyn ExclusiveStorage, Option<u32>),
    reset: impl FnOnce(u32, u32) -> Result<()>,
) -> Result<()> {
    let from = segments.released_from;
    let count = limit.map_or(segments.count(), |limit| {
        from.saturating_add(limit).min(segments.count())
    });
    if from == count {
        return Ok(());
    }
    let first_unit = segments.first_unit(from);
    let units = segments.first_unit(count) - first_unit;

    let mut restored = 0;
    let mut failure = None;
    for (index, storage) in storages.iter().enumerate() {
        let config = segmented_config(device, segments, index);
        let indices = segments.parts_of(index, from..count);
        if let Err(error) = owner.restore_segments(&config, storage, &indices) {
            failure = Some(error);
            break;
        }
        restored += 1;
    }

    // The new segments are mapped, so their units are prepared before anything reads them.
    if failure.is_none() {
        segments.released_from = count;
        if let Err(error) = reset(first_unit, units) {
            segments.released_from = from;
            failure = Some(error);
        }
    }

    if let Some(error) = failure {
        // Put the set back as asleep as it was: the segments this call created go back through
        // the owner that created them, so its reservation ends with them.
        for (index, storage) in storages.iter().enumerate().take(restored) {
            let config = segmented_config(device, segments, index);
            let indices = segments.parts_of(index, from..count);
            owner
                .release_segments(&config, storage, &indices)
                .map_err(|undo| {
                    Error::Invariant(format!(
                        "a wake failed ({error}), and giving the {restored} storage(s) it \
                         restored back failed too ({undo}): the set is half woken and its next \
                         wake would ask for segments that exist"
                    ))
                })?;
        }
        return Err(error);
    }

    tracing::info!(
        names = ?segments.storages.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        first = from,
        segments = count - from,
        "created the segments again"
    );
    Ok(())
}

/// Checks a pool of `units` units of `pool`: the padding unit and one usable unit at least, and
/// planes of bytes.
fn check(pool: &PoolLayout, units: u32) -> Result<()> {
    if units < 2 {
        return Err(Error::Invalid(
            "a cache unit pool needs the padding unit and one usable unit".into(),
        ));
    }
    if pool.fields.is_empty() || pool.segment_plane() == 0 {
        return Err(Error::Invalid(
            "a cache unit pool needs fields and planes of bytes".into(),
        ));
    }
    Ok(())
}

/// The runs of the exclusive storage of `config`, columns of `units` cells of `cells[c]` bytes
/// each, back to back: one run of every unit per column, named `{name}/{column}`. Empty for
/// storage local to the process.
fn cell_runs(
    owner: &dyn ExclusiveStorage,
    config: &ExclusiveConfig<'_>,
    cells: &[u64],
    units: u32,
) -> Result<Vec<crate::memory::cache::info::CachePlaneRun>> {
    let Some((key, gpu)) = owner.key(config)? else {
        return Ok(Vec::new());
    };

    let mut offset = 0;
    Ok(cells
        .iter()
        .enumerate()
        .map(|(column, &bytes)| {
            let run = crate::memory::cache::info::CachePlaneRun {
                field: ::storage::protocol::planes::field(config.name, column as u32),
                key,
                gpu,
                first: 0,
                count: units,
                offset,
                bytes,
            };
            offset += u64::from(units) * bytes;
            run
        })
        .collect())
}

/// What [`take_split`] takes: the split storages in the order they were named, the segments'
/// state, and the segments created for this rank.
type Split = (Vec<Arc<Storage>>, Segments, Vec<u32>);

/// Takes the split storages `storages` of a set in `parts` segments on `device`, which `peers`
/// address, from `owner`, adding their runs to `runs`: each storage holds a column per entry of
/// its bytes, the part of a column of a segment its units at that column's bytes each (a field's
/// column holds a plane per unit; the state units are one column of a checkpoint per unit).
/// Returns the storages, the segments' state and the segments created for this rank.
///
/// # Errors
///
/// When `owner` refuses a storage, or the segments given back are not the same ones of every
/// column of every storage, from one segment on.
pub(super) fn take_split(
    storages: &[SplitStorage],
    parts: &[PlaneSegment],
    (device, peers): (DeviceId, &[DeviceId]),
    owner: &dyn ExclusiveStorage,
    runs: &mut Vec<crate::memory::cache::info::CachePlaneRun>,
) -> Result<Split> {
    let count = parts.len() as u32;

    let mut taken = Vec::with_capacity(storages.len());
    let mut kept = Vec::with_capacity(storages.len());
    let mut created = vec![false; parts.len()];
    for (name, columns) in storages {
        // Part `c · count + k` holds column `c` of segment `k`: its units at the column's bytes
        // each.
        let segment_parts: Vec<SegmentPart> = columns
            .iter()
            .enumerate()
            .flat_map(|(column, &unit_bytes)| {
                parts.iter().map(move |part| SegmentPart {
                    length: part.segment.units as usize * unit_bytes as usize,
                    column: column as u32,
                    segment: part.segment,
                    create: part.create,
                })
            })
            .collect();
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
                    field: ::storage::protocol::planes::field(name, part.column),
                    key,
                    gpu,
                    first,
                    count: part.segment.units,
                    offset: 0,
                    bytes: columns[part.column as usize],
                });
            }
            first += part.segment.units;

            created[index % parts.len()] |= made.get(index).copied().unwrap_or(false);
        }
        taken.push(storage);
        kept.push((name.clone(), segment_parts));
    }

    // Segments given back, which no rank of the region holds, are the same ones of every
    // column of every split storage, from one segment on.
    let mapped = |segment: u32| -> Result<Vec<bool>> {
        let mut mapped = Vec::new();
        for (storage, (_, parts)) in taken.iter().zip(&kept) {
            let columns = parts.len() as u32 / count.max(1);
            for column in 0..columns {
                mapped.push(storage.segment_mapped((column * count + segment) as usize)?);
            }
        }
        Ok(mapped)
    };
    let mut released_from = count;
    for segment in 0..count {
        if mapped(segment)?.iter().all(|mapped| !mapped) {
            released_from = segment;
            break;
        }
    }
    for segment in 0..count {
        let gone = mapped(segment)?.iter().any(|mapped| !mapped);
        if gone != (segment >= released_from) {
            return Err(Error::Invalid(format!(
                "segment {segment} of {:?} is {} while the set is given back from segment \
                 {released_from}",
                storages.iter().map(|(name, _)| name).collect::<Vec<_>>(),
                if gone { "unmapped" } else { "mapped" }
            )));
        }
    }

    let segments = Segments {
        segments: parts.iter().map(|part| part.segment).collect(),
        released_from,
        storages: kept,
    };
    let created = (0..count)
        .filter(|segment| created[*segment as usize])
        .collect();

    Ok((taken, segments, created))
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
    use nn::family::{CacheField, FieldOwner, Format, History, INDEX_KEY, KEY, Record, VALUE};
    use proto::request::PrefixCacheDtype;

    use super::super::layout::CacheLayout;
    use super::*;

    /// The keys and values of `layers` layers of one 8-wide head.
    fn keys_and_values(layers: u32) -> Vec<CacheField> {
        CacheField::keys_and_values(layers, 1, 8)
    }

    /// Index keys of six elements per four tokens in BF16 over `layers` layers.
    fn index_keys(layers: u32) -> CacheField {
        CacheField {
            name: INDEX_KEY,
            owners: (0..layers)
                .map(|layer| FieldOwner { layer, ratio: 4 })
                .collect(),
            readers: Vec::new(),
            history: History::Full,
            record: Record::rows(Format::Elements(DType::BF16), 1, 6),
        }
    }

    /// The KV pool of `fields` over 16-token pages.
    fn layout(fields: &[CacheField], dtype: PrefixCacheDtype) -> CacheLayout {
        CacheLayout::of(16, fields, dtype).expect("a layout of the fields")
    }

    /// Planes of two layers of 16-token pages of one 8-wide head, over eight units, with no
    /// device.
    fn planes(dtype: PrefixCacheDtype) -> (CacheLayout, Planes) {
        let layout = layout(&keys_and_values(2), dtype);
        let planes = Planes::meta(layout.kv().clone(), 8).expect("planes of eight units");
        (layout, planes)
    }

    #[test]
    fn runs_of_units_are_consecutive_and_each_unit_counts_once() {
        assert_eq!(unit_runs(&[5, 3, 4, 9, 4]), vec![(3, 3), (9, 1)]);
        assert!(unit_runs(&[]).is_empty());
    }

    /// The storages of the fields of keys and values, and of index keys, are the ones a handoff
    /// between replicas reads a rank's fields by.
    #[test]
    fn the_storages_of_keys_values_and_index_keys_keep_the_names_a_handoff_reads() {
        use ::storage::protocol::planes::{FLAGS, INDEX_KEYS, SCALES, VALUES};

        for (plane, name) in [KEY, VALUE].into_iter().enumerate() {
            let [values, flags, scales] = storage_names(name);
            assert_eq!(
                (values.as_str(), flags.as_str(), scales.as_str()),
                (VALUES[plane], FLAGS[plane], SCALES[plane])
            );
        }
        assert_eq!(storage_names(INDEX_KEY)[0], INDEX_KEYS);
    }

    /// A layer's planes are its column's units, and a dense cache states no scale.
    #[test]
    fn a_layer_is_one_column_of_every_unit() -> Result<()> {
        let (layout, dense) = planes(PrefixCacheDtype::BFloat16);
        let lent = dense.planes(&layout.groups[0])?;

        // Keys of layers 0 and 1, then their values.
        let fields: Vec<(&str, u32)> = lent
            .iter()
            .map(|plane| (plane.field, plane.layer))
            .collect();
        assert_eq!(fields, [(KEY, 0), (KEY, 1), (VALUE, 0), (VALUE, 1)]);
        let key = &lent[1];
        assert_eq!(key.entries.dims(), [8, 16, 1, 8]);
        assert_eq!(lent[3].entries.dtype(), DType::BF16);
        assert_eq!(key.initialized.dims(), [8, 1, 1, 1]);
        assert_eq!(key.initialized.dtype(), DType::U8);
        assert!(lent.iter().all(|plane| plane.scale.is_none()));

        let (layout, fp8) = planes(PrefixCacheDtype::Float8E4m3Fn);
        let key = &fp8.planes(&layout.groups[0])?[0];
        assert_eq!(key.entries.dtype(), DType::F8E4M3);
        let scale = key.scale.as_ref().expect("an FP8 cache states its scales");
        assert_eq!(
            (scale.dims(), scale.dtype()),
            (&[8, 1, 1, 1][..], DType::F32)
        );
        Ok(())
    }

    /// A layer of a cache that keeps index keys lends them on its pages: one key of the stated
    /// width per `ratio` tokens of each unit of its column, in their own element type whatever
    /// the keys and values are stored in.
    #[test]
    fn a_layer_lends_the_index_keys_of_its_pages() -> Result<()> {
        for dtype in [PrefixCacheDtype::BFloat16, PrefixCacheDtype::Float8E4m3Fn] {
            let mut fields = keys_and_values(2);
            fields.push(index_keys(2));
            let layout = layout(&fields, dtype);
            let planes = Planes::meta(layout.kv().clone(), 8)?.planes(&layout.groups[0])?;

            let index = planes
                .iter()
                .find(|plane| plane.field == INDEX_KEY && plane.layer == 1)
                .expect("the layer keeps index keys");
            assert_eq!(
                (index.entries.dims(), index.entries.dtype(), index.ratio),
                (&[8, 4, 1, 6][..], DType::BF16, 4)
            );
            assert!(index.scale.is_none());
        }
        Ok(())
    }

    /// A draft layer's planes lie over the columns its pages do, its index keys included: one
    /// page per cell.
    #[test]
    fn a_draft_layer_lends_the_planes_of_its_span() -> Result<()> {
        let mut fields = keys_and_values(4);
        fields.push(index_keys(4));
        let mut drafter = keys_and_values(1);
        drafter.push(index_keys(1));
        let layout = layout(&fields, PrefixCacheDtype::BFloat16).with_draft(&drafter)?;
        let draft = layout.groups[1].clone();
        let span = layout.draft_span().expect("the layout holds a draft group");
        assert_eq!(span, 4);

        let planes = Planes::meta(layout.kv().clone(), 8)?.draft_planes(&draft, span)?;
        let fields: Vec<&str> = planes.iter().map(|plane| plane.field).collect();
        assert_eq!(fields, [KEY, VALUE, INDEX_KEY]);
        assert_eq!(planes[0].entries.dims(), [32, 16, 1, 8]);
        assert_eq!(planes[2].entries.dims(), [32, 4, 1, 6]);
        assert!(planes.iter().all(|plane| plane.layer == 0));
        Ok(())
    }

    /// Owners of one field with other ratios lie in columns of other planes, back to back: a
    /// plane's offset is every column before it, whole.
    #[test]
    fn owners_of_two_ratios_lie_in_columns_of_their_own_planes() -> Result<()> {
        let latent = CacheField {
            name: "compressed_latent",
            owners: vec![
                FieldOwner { layer: 2, ratio: 2 },
                FieldOwner {
                    layer: 20,
                    ratio: 1,
                },
            ],
            readers: Vec::new(),
            history: History::Full,
            record: Record::rows(Format::Nvfp4, 1, 512),
        };
        let layout = layout(&[latent], PrefixCacheDtype::BFloat16);
        let pool = layout.kv();
        let field = &pool.fields[0];

        // Eight entries of 288 bytes, then sixteen: a unit of 6912 bytes of planes.
        assert_eq!(field.plane_bytes(0, 16), 8 * 288);
        assert_eq!(field.plane_bytes(1, 16), 16 * 288);
        assert_eq!(pool.split_bytes(), 24 * 288);
        assert_eq!(field.column_offset(1, 10, 16), 10 * 8 * 288);
        assert_eq!(pool.plane_bytes(), 8 * 288);

        let planes = Planes::meta(pool.clone(), 10)?.planes(&layout.groups[0])?;
        assert_eq!(planes[0].entries.dims(), [10, 8, 1, 288]);
        assert_eq!(planes[1].entries.dims(), [10, 16, 1, 288]);
        assert_eq!((planes[1].layer, planes[1].ratio), (20, 1));
        Ok(())
    }

    /// A refused copy names the pair and the reason.
    #[test]
    fn a_refused_unit_copy_names_the_pair_and_the_reason() {
        let (_, planes) = planes(PrefixCacheDtype::BFloat16);
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
        let (_, mut planes) = planes(PrefixCacheDtype::BFloat16);

        assert_eq!(planes.mapped_bytes(), 0);
        assert!(planes.segments().is_none());
        assert_eq!(planes.held_segments(), 0);
        assert!(planes.runs().is_empty());
        assert!(planes.plane_address(0, 0, 1).is_err());
        assert!(planes.flag_address(1, 0, 1).is_err());
        assert!(
            planes
                .release_segments(0, &crate::memory::exclusive::LocalStorage)
                .is_err()
        );
    }

    /// A pool needs the padding unit and one unit a page can take.
    #[test]
    fn a_pool_of_one_unit_is_refused() {
        let layout = layout(&keys_and_values(2), PrefixCacheDtype::BFloat16);
        assert!(Planes::meta(layout.kv().clone(), 1).is_err());
    }
}
