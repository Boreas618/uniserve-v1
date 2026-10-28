//! The cache unit pool's device storage and its unit-level resets.
//!
//! Each field (key, value) owns separate allocations in the layout the
//! numerical cache library reads (`uniserve.prefix_cache.units`):
//!
//! | Field | Element | Shape |
//! | --- | --- | --- |
//! | `{key,value}.values` | `u8` | `[columns, num_units, plane_bytes]` |
//! | `{key,value}.scale` (FP8 only) | `f32` | `[columns, num_units]` |
//! | `{key,value}.initialized` | `bool` | `[columns, num_units]` |
//!
//! so the byte layout is `[field][column][unit][token][head][dim]`: layer
//! `k` of a group lives in column `k % columns` of its page's unit
//! `k / columns`, and a homogeneous model's planes equal the classic
//! `[layers, blocks, block_size, heads, dim]` stack.
//!
//! Scattered unit resets (recycle, zero, mark-initialized) are bound Python
//! functions over a Rust-owned device column of unit ids: an indexed fill
//! costs a constant few kernels however the units are scattered, while
//! native fills would cost one or more driver calls per run of consecutive
//! units. Contiguous resets (startup scratch) are native fills.
//!
//! On a device, each value plane is made of segments along the units, each
//! column's part of a segment one allocation, mapped back to back so the
//! plane keeps its layout and addresses ([`PlaneLayout`]). A world's own
//! units come first, split into at most [`SEGMENTS`] segments; a world that
//! succeeds another on the same devices maps, after them, the segments its
//! predecessor's worker of the same scope holds, read-only, so it reads the
//! predecessor's pages and cannot change them, and takes them once the
//! predecessor is gone ([`CachePlanes::take_inherited`]). The last segments
//! are given back to the device when a successor world needs their memory,
//! or all of them while the deployment sleeps
//! ([`CachePlanes::release_segments`]), and created again at their
//! addresses after ([`CachePlanes::restore_segments`]); the engine allocates
//! none of their units meanwhile.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use cuda::{CopyBatch, DeviceId, SourceOrder, Stream};
use tvm_ffi::Any;

use super::plan::CachePlan;
use crate::ffi::{
    Backing, DType, DeferredCall, Function, Handle, Interpreter, MemoryKind, strings,
};
use crate::memory::allocation::{Storage, StorageView};
use crate::memory::retained::{
    ExclusiveMemory, ExclusiveSpec, Retained, SegmentPart, SegmentedSpec,
};
use crate::stream::{Issue, RowFill};
use crate::{Error, Result};

/// The two K/V fields, in storage order.
pub(crate) const FIELDS: [&str; 2] = ["key", "value"];

/// The bit pattern of `1.0f32`, the neutral FP8 scale.
const UNIT_SCALE: u32 = 0x3F80_0000;

/// The segments a device value plane is made of along its units: the
/// granularity in which the cache gives memory back, an eighth of it.
pub(crate) const SEGMENTS: u32 = 8;

/// How a pool of `num_units` units splits its value planes into segments
/// whose every column's part is whole allocation granules of
/// `granularity` bytes, with planes of `plane_bytes` per unit: the segments'
/// count and the units each holds. `None` when the pool is too small to
/// split, or not a multiple of the split ([`segmented_units`] rounds it).
pub(crate) fn segment_layout(
    num_units: u32,
    plane_bytes: u64,
    granularity: u64,
) -> Option<(u32, u32)> {
    let granule = granule_units(plane_bytes, granularity)?;
    let segments = SEGMENTS.min(num_units / granule);
    if segments == 0 {
        return None;
    }
    let units = num_units / segments / granule * granule;
    (units * segments == num_units).then_some((segments, units))
}

/// `num_units` rounded down to the largest pool whose planes split into
/// segments ([`segment_layout`]); `num_units` itself when it is too small to
/// split.
pub(crate) fn segmented_units(num_units: u32, plane_bytes: u64, granularity: u64) -> u32 {
    let Some(granule) = granule_units(plane_bytes, granularity) else {
        return num_units;
    };
    let segments = SEGMENTS.min(num_units / granule);
    if segments == 0 {
        return num_units;
    }
    num_units / segments / granule * granule * segments
}

/// The fewest units whose planes of `plane_bytes` fill whole granules.
fn granule_units(plane_bytes: u64, granularity: u64) -> Option<u32> {
    if plane_bytes == 0 || granularity == 0 {
        return None;
    }
    let (mut a, mut b) = (plane_bytes, granularity);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    u32::try_from(granularity / a).ok()
}

/// How a device pool's value planes are laid out along its units: its
/// segments in unit order, each of whole allocation granules on every
/// column, and which of them the predecessor world's worker writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneLayout {
    pub(crate) segments: Vec<LaidSegment>,
}

/// One segment of a [`PlaneLayout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaidSegment {
    pub(crate) segment: ipc::CacheSegment,
    /// Whether the predecessor world's worker writes the segment: it is
    /// mapped read-only until the world's commit.
    pub(crate) inherited: bool,
    /// Whether the planes create the segment when its memory does not exist:
    /// a successor's tail segments that its memory does not hold yet are
    /// left absent, given back from the start.
    pub(crate) create: bool,
}

impl PlaneLayout {
    /// A pool of `num_units` units that `world` creates whole: its units
    /// split into at most [`SEGMENTS`] equal segments ([`segment_layout`]),
    /// numbered from zero. `None` when the pool does not split.
    pub(crate) fn own(
        world: u64,
        num_units: u32,
        plane_bytes: u64,
        granularity: u64,
    ) -> Option<Self> {
        let (count, units) = segment_layout(num_units, plane_bytes, granularity)?;
        Some(Self {
            segments: (0..count)
                .map(|ordinal| LaidSegment {
                    segment: ipc::CacheSegment {
                        world,
                        ordinal,
                        units,
                    },
                    inherited: false,
                    create: true,
                })
                .collect(),
        })
    }

    /// The pool of a world that succeeds `predecessor`: a head segment of
    /// its own of `head_units` (the sentinel and the warmup scratch), the
    /// predecessor's segments, read-only, then `tail_units` of its own in
    /// at most [`SEGMENTS`] equal segments, which the world creates as its
    /// memory frees ([`CachePlanes::restore_segments`]) and takes in full
    /// once its predecessor is gone.
    ///
    /// # Errors
    ///
    /// Fails when the head does not fill whole granules.
    pub(crate) fn successor(
        world: u64,
        head_units: u32,
        tail_units: u32,
        plane_bytes: u64,
        granularity: u64,
        predecessor: &ipc::PredecessorCache,
    ) -> Result<Self> {
        let granule = granule_units(plane_bytes, granularity).unwrap_or(1);
        if head_units == 0 || !head_units.is_multiple_of(granule) {
            return Err(Error::Invalid(format!(
                "a head segment of {head_units} cache units fills no whole granules of {granule}"
            )));
        }
        let mut segments = vec![LaidSegment {
            segment: ipc::CacheSegment {
                world,
                ordinal: 0,
                units: head_units,
            },
            inherited: false,
            create: true,
        }];
        segments.extend(predecessor.segments.iter().map(|segment| LaidSegment {
            segment: *segment,
            inherited: true,
            create: false,
        }));
        if let Some((count, units)) = segment_layout(tail_units, plane_bytes, granularity) {
            segments.extend((0..count).map(|index| LaidSegment {
                segment: ipc::CacheSegment {
                    world,
                    ordinal: 1 + index,
                    units,
                },
                inherited: false,
                create: true,
            }));
        }
        Ok(Self { segments })
    }

    /// Leaves the tail segments past the first `count` of them to be
    /// created once the world's memory holds them.
    pub(crate) fn defer_tail(&mut self, count: usize) {
        for laid in self
            .segments
            .iter_mut()
            .skip_while(|laid| laid.segment.ordinal == 0 || laid.inherited)
            .skip(count)
        {
            laid.create = false;
        }
    }

    /// The units and segments of the tail the world creates at once.
    pub(crate) fn tail_segment_units(&self) -> Option<u32> {
        self.segments
            .iter()
            .find(|laid| !laid.inherited && laid.segment.ordinal > 0)
            .map(|laid| laid.segment.units)
    }

    /// The fewest units of whole granules that hold `units`.
    pub(crate) fn whole_granules(units: u32, plane_bytes: u64, granularity: u64) -> u32 {
        let granule = granule_units(plane_bytes, granularity).unwrap_or(1).max(1);
        units.div_ceil(granule).max(1) * granule
    }

    /// The exact segments a rank's planes were made of: as the serving
    /// worker of a standby rank group's scope holds them, or as a world's
    /// rank held them before its relaunch; its own world's, none inherited.
    /// The segments past the ones held are left absent.
    pub(crate) fn adopted(table: &ipc::CacheTable) -> Self {
        Self {
            segments: table
                .segments
                .iter()
                .enumerate()
                .map(|(index, segment)| LaidSegment {
                    segment: *segment,
                    inherited: false,
                    create: (index as u32) < table.held,
                })
                .collect(),
        }
    }

    /// The units the planes hold.
    pub(crate) fn units(&self) -> u32 {
        self.segments.iter().map(|laid| laid.segment.units).sum()
    }

    /// The units of the segments the predecessor writes:
    /// `[start, start + units)`.
    pub(crate) fn inherited(&self) -> std::ops::Range<u32> {
        let mut first = 0;
        let mut units = 0;
        for laid in &self.segments {
            if laid.inherited {
                if units == 0 {
                    first = self.first_unit(laid);
                }
                units += laid.segment.units;
            }
        }
        first..first + units
    }

    fn first_unit(&self, target: &LaidSegment) -> u32 {
        self.segments
            .iter()
            .take_while(|laid| !std::ptr::eq(*laid, target))
            .map(|laid| laid.segment.units)
            .sum()
    }
}

/// The segments of a device pool's value planes and the first one given
/// back (`segments.len()` while none is).
struct Segmenting {
    /// The segments in unit order, each with whether the predecessor world
    /// still writes it.
    segments: Vec<(ipc::CacheSegment, AtomicBool)>,
    released_from: AtomicU32,
    /// Each value plane's parts, column-major: the part of segment `k` that
    /// column `c` holds is `c * segments.len() + k`.
    parts: Vec<SegmentPart>,
}

impl Segmenting {
    fn count(&self) -> u32 {
        self.segments.len() as u32
    }

    /// The first unit of segment `index`.
    fn first_unit(&self, index: u32) -> u32 {
        self.segments
            .iter()
            .take(index as usize)
            .map(|(segment, _)| segment.units)
            .sum()
    }

    /// The parts of segments `range`, every column's.
    fn parts_of(&self, range: std::ops::Range<u32>) -> Vec<usize> {
        let count = self.count();
        (0..self.parts.len() as u32 / count.max(1))
            .flat_map(|column| {
                range
                    .clone()
                    .map(move |segment| (column * count + segment) as usize)
            })
            .collect()
    }
}
/// The pool's storage on one device (or host memory for computations on the
/// host).
pub struct CachePlanes {
    plan: Arc<CachePlan>,
    device: Option<DeviceId>,
    num_units: u32,
    /// The value planes' segments, on a device whose pool splits into them.
    segmenting: Option<Segmenting>,
    values: [Arc<Storage>; 2],
    scales: Option<[Arc<Storage>; 2]>,
    initialized: [Arc<Storage>; 2],
    /// Each field's whole view, keyed as the numerical library reads it.
    /// Every binding passes these views, so each field is wrapped as a
    /// torch tensor once.
    fields: Vec<(String, StorageView)>,
}

impl CachePlanes {
    /// One plane cell as a borrowed region of its retained allocation.
    /// The caller keeps the scheduler's page reference until every reader
    /// of this region finishes; a descriptor alone does not pin a page.
    pub(crate) fn retained_region(
        &self,
        retained: &Retained,
        field: usize,
        column: u32,
        unit: u32,
        suffix: &str,
    ) -> Result<storage::manifest::Chunk> {
        let device = self
            .device
            .ok_or_else(|| Error::Invalid("cache transfers require a device pool".into()))?;
        let gpu = device.uuid()?.0;
        let name = format!("prefix_cache.{}.{suffix}", FIELDS[field]);
        let bytes = match suffix {
            "values" => self.plan.plane_bytes,
            "scale" => 4,
            "initialized" => 1,
            _ => return Err(Error::Invalid(format!("unknown cache field {suffix}"))),
        };
        let (key, offset) = if suffix == "values"
            && let Some(segmented) = &self.segmenting
        {
            let mut first = 0;
            let segment = segmented
                .segments
                .iter()
                .enumerate()
                .find_map(|(index, (segment, _))| {
                    let start = first;
                    first += segment.units;
                    (unit < first).then_some((index, start))
                })
                .ok_or_else(|| Error::Invalid(format!("cache unit {unit} is outside its pool")))?;
            if segment.0 as u32 >= segmented.released_from.load(Ordering::Acquire) {
                return Err(Error::Invalid(format!(
                    "cache unit {unit} has no retained storage"
                )));
            }
            let part = &segmented.parts[column as usize * segmented.segments.len() + segment.0];
            (
                retained.segment_key(&name, part, &gpu),
                u64::from(unit - segment.1) * bytes,
            )
        } else {
            let spec = ExclusiveSpec {
                name: &name,
                device,
                bytes: (u64::from(self.plan.columns) * u64::from(self.num_units) * bytes) as usize,
                access: &[],
            };
            (
                retained.exclusive_key(&spec, &gpu),
                (u64::from(column) * u64::from(self.num_units) + u64::from(unit)) * bytes,
            )
        };
        Ok(storage::manifest::Chunk {
            offset: 0,
            length: bytes,
            key,
            storage_offset: offset,
            binding: storage::manifest::Binding::Rank,
        })
    }

    /// Allocate a pool of `num_units` units (the sentinel included), and
    /// reset the planes whose memory is new: values and flags zero, scales
    /// one. On a device the resets run on `stream`, which is synchronized
    /// before returning.
    ///
    /// Device planes are exclusive storages `memory` provides, whose
    /// contents outlive the worker. Planes a predecessor left keep their
    /// contents, which are finite whatever they hold: a standby worker's
    /// predecessor serves from them, a successor engine reads the pages its
    /// predecessor handed over, and every unit is recycled for its next
    /// owner.
    ///
    /// Every local peer device may address the values, because readers on
    /// those devices in this process copy from the pages directly; other
    /// ranks read a frozen copy of the pages of segmented planes
    /// (`cache_export`).
    pub fn allocate(
        plan: Arc<CachePlan>,
        device: Option<DeviceId>,
        num_units: u32,
        layout: Option<&PlaneLayout>,
        stream: Option<&Stream>,
        memory: &mut dyn ExclusiveMemory,
    ) -> Result<Self> {
        if layout.is_some_and(|layout| layout.units() != num_units) {
            return Err(Error::Invalid(
                "a cache pool's segments do not hold its units".into(),
            ));
        }
        if num_units < 2 {
            return Err(Error::Invalid(
                "a cache unit pool needs the sentinel and one usable unit".into(),
            ));
        }
        if device.is_some() != stream.is_some() {
            return Err(Error::Invalid(
                "device planes are reset on a stream of their device".into(),
            ));
        }
        let cells = plan.columns as usize * num_units as usize;
        let plane_bytes = cells * plan.plane_bytes as usize;
        // Whether each field's storages (values, flags, scales) are new.
        let mut created = [true; 3];
        let mut segmenting = None;
        // Segments created beside ones a predecessor left: their units are
        // reset alone.
        let mut fresh_segments: Vec<u32> = Vec::new();
        let (values, initialized, scales) = match device {
            Some(device) => {
                let peers = Storage::peers(device)?;
                let spec = |name, bytes| ExclusiveSpec {
                    name,
                    device,
                    bytes,
                    access: &peers,
                };
                let mut specs = Vec::new();
                if layout.is_none() {
                    specs.push(spec("prefix_cache.key.values", plane_bytes));
                    specs.push(spec("prefix_cache.value.values", plane_bytes));
                }
                specs.push(spec("prefix_cache.key.initialized", cells));
                specs.push(spec("prefix_cache.value.initialized", cells));
                if plan.quantized {
                    specs.push(spec("prefix_cache.key.scale", cells * 4));
                    specs.push(spec("prefix_cache.value.scale", cells * 4));
                }
                let taken = memory.exclusive(&specs)?;
                let mut taken = taken.into_iter();
                let mut next = || {
                    taken
                        .next()
                        .ok_or_else(|| Error::Invariant("an exclusive storage is missing".into()))
                };
                let values = match layout {
                    None => {
                        let (key, value) = (next()?, next()?);
                        created[0] = key.created || value.created;
                        [key.storage, value.storage]
                    }
                    Some(layout) => {
                        let count = layout.segments.len() as u32;
                        let plane_bytes = plan.plane_bytes as usize;
                        let parts: Vec<SegmentPart> = (0..plan.columns)
                            .flat_map(|column| {
                                layout.segments.iter().map(move |laid| SegmentPart {
                                    length: laid.segment.units as usize * plane_bytes,
                                    column,
                                    segment: laid.segment,
                                    inherited: laid.inherited,
                                    create: laid.create,
                                })
                            })
                            .collect();
                        let mut take = |name| {
                            memory.segmented(&SegmentedSpec {
                                name,
                                device,
                                parts: &parts,
                                access: &peers,
                            })
                        };
                        let key = take("prefix_cache.key.values")?;
                        let value = take("prefix_cache.value.values")?;
                        // Segments created for this pool, of every column;
                        // all of them make the values new as a whole.
                        let made: Vec<u32> = (0..count)
                            .filter(|segment| {
                                (0..plan.columns).any(|column| {
                                    let index = (column * count + segment) as usize;
                                    key.created[index] || value.created[index]
                                })
                            })
                            .collect();
                        created[0] = made.len() == count as usize;
                        if !created[0] {
                            fresh_segments = made;
                        }
                        // A standby finds none of the segments its
                        // predecessor gave back: the same ones of every
                        // column, from one segment on.
                        let absent = |segment: u32| {
                            (0..plan.columns).all(|column| {
                                let index = (column * count + segment) as usize;
                                key.absent[index] && value.absent[index]
                            })
                        };
                        let released_from =
                            (0..count).find(|segment| absent(*segment)).unwrap_or(count);
                        for column in 0..plan.columns {
                            for segment in 0..count {
                                let index = (column * count + segment) as usize;
                                let gone = key.absent[index] || value.absent[index];
                                if gone != (segment >= released_from) {
                                    return Err(Error::Invalid(format!(
                                        "the cache planes' segment {segment} of column {column} is \
                                         {} while the planes are given back from segment {released_from}",
                                        if gone { "absent" } else { "present" }
                                    )));
                                }
                            }
                        }
                        segmenting = Some(Segmenting {
                            segments: layout
                                .segments
                                .iter()
                                .map(|laid| (laid.segment, AtomicBool::new(laid.inherited)))
                                .collect(),
                            released_from: AtomicU32::new(released_from),
                            parts,
                        });
                        [key.storage, value.storage]
                    }
                };
                let (key, value) = (next()?, next()?);
                created[1] = key.created || value.created;
                let initialized = [key.storage, value.storage];
                let scales = if plan.quantized {
                    let (key, value) = (next()?, next()?);
                    created[2] = key.created || value.created;
                    Some([key.storage, value.storage])
                } else {
                    None
                };
                (values, initialized, scales)
            }
            None => (
                [Storage::host(plane_bytes), Storage::host(plane_bytes)],
                [Storage::host(cells), Storage::host(cells)],
                plan.quantized
                    .then(|| [Storage::host(cells * 4), Storage::host(cells * 4)]),
            ),
        };
        let mut planes = Self {
            plan,
            device,
            num_units,
            segmenting,
            values,
            scales,
            initialized,
            fields: Vec::new(),
        };
        planes.fields = planes.field_views()?;

        // One row spanning each allocation.
        let whole = |address: u64, value: u32, width: usize, count: usize| RowFill {
            address,
            pitch: count * width,
            value,
            width,
            count,
            height: 1,
        };
        let [values_created, flags_created, scales_created] = created;
        for field in 0..2 {
            // SAFETY: each allocation spans exactly the filled bytes and is
            // owned by `planes`; the stream is drained below.
            unsafe {
                if values_created {
                    stream.memset_2d(whole(planes.values[field].address(), 0, 1, plane_bytes))?;
                }
                if flags_created {
                    stream.memset_2d(whole(planes.initialized[field].address(), 0, 1, cells))?;
                }
                if let Some(scales) = &planes.scales
                    && scales_created
                {
                    stream.memset_2d(whole(scales[field].address(), UNIT_SCALE, 4, cells))?;
                }
            }
        }
        if let (Some(segmenting), Some(stream)) = (&planes.segmenting, stream) {
            for segment in fresh_segments {
                let units = segmenting.segments[segment as usize].0.units;
                planes.reset_units(segmenting.first_unit(segment), units, true, stream)?;
            }
        }
        if let Some(stream) = stream {
            stream.synchronize()?;
        }
        Ok(planes)
    }

    /// The segments the value planes are made of, in unit order; `None` for
    /// planes that are not split.
    pub(crate) fn segments(&self) -> Option<Vec<ipc::CacheSegment>> {
        self.segmenting.as_ref().map(|segmenting| {
            segmenting
                .segments
                .iter()
                .map(|(segment, _)| *segment)
                .collect()
        })
    }

    /// The segments the planes hold, `[0, held)`: the first given back.
    pub(crate) fn held_segments(&self) -> u32 {
        self.segmenting.as_ref().map_or(0, |segmenting| {
            segmenting.released_from.load(Ordering::Acquire)
        })
    }

    /// The units the planes map read-only from the predecessor world's
    /// planes; empty once the world took them, or for planes of its own.
    pub(crate) fn inherited(&self) -> std::ops::Range<u32> {
        let Some(segmenting) = &self.segmenting else {
            return 0..0;
        };
        let mut range: Option<std::ops::Range<u32>> = None;
        for (index, (segment, inherited)) in segmenting.segments.iter().enumerate() {
            if inherited.load(Ordering::Acquire) {
                let first = segmenting.first_unit(index as u32);
                range = Some(match range {
                    Some(range) => range.start..first + segment.units,
                    None => first..first + segment.units,
                });
            }
        }
        range.unwrap_or(0..0)
    }

    /// The device bytes one unit of every value plane and column holds.
    pub(crate) fn unit_bytes(&self) -> u64 {
        2 * u64::from(self.plan.columns) * self.plan.plane_bytes
    }

    /// Takes the segments mapped read-only from the predecessor world, whose
    /// writer is gone, through `retained`: they become writable at unchanged
    /// addresses and the world's own. Returns how many it took.
    pub(crate) fn take_inherited(
        &self,
        retained: &crate::memory::retained::Retained,
    ) -> Result<usize> {
        let taken = retained.take_inherited()?;
        if let Some(segmenting) = &self.segmenting {
            for (_, inherited) in &segmenting.segments {
                inherited.store(false, Ordering::Release);
            }
        }
        Ok(taken)
    }

    /// Copies the predecessor's cache flags and scales of the units the
    /// planes inherit from it into the planes' own, issued to `stream`,
    /// which is synchronized before returning: the predecessor's planes
    /// hold the flags and scales of `predecessor_units` units in
    /// `predecessor` (each field's flags, then its scales when quantized),
    /// and the predecessor parked, so they no longer change.
    pub(crate) fn adopt_inherited(
        &self,
        predecessor: &[Arc<Storage>],
        predecessor_units: u32,
        stream: &Stream,
    ) -> Result<()> {
        let inherited = self.inherited();
        if inherited.is_empty() {
            return Ok(());
        }
        let per_field = if self.scales.is_some() { 2 } else { 1 };
        if predecessor.len() != 2 * per_field {
            return Err(Error::Invariant(
                "a predecessor's cache flags and scales are missing".into(),
            ));
        }
        let mut batch = CopyBatch::new(SourceOrder::DuringCall);
        for field in 0..2 {
            let mut targets = vec![(&self.initialized[field], 1_u64)];
            if let Some(scales) = &self.scales {
                targets.push((&scales[field], 4));
            }
            for (offset, (target, width)) in targets.into_iter().enumerate() {
                let source = &predecessor[field * per_field + offset];
                for column in 0..self.plan.columns {
                    let from = (u64::from(column) * u64::from(predecessor_units)) * width;
                    let to = (u64::from(column) * u64::from(self.num_units)
                        + u64::from(inherited.start))
                        * width;
                    batch.push(
                        target.address() + to,
                        source.address() + from,
                        (u64::from(inherited.end - inherited.start) * width) as usize,
                    );
                }
            }
        }
        // SAFETY: every range lies inside its storage: the predecessor's
        // planes hold `predecessor_units` units of every column and the
        // inherited ones are their leading units; the stream is drained.
        unsafe { stream.copy(&batch) }?;
        stream.synchronize()?;
        Ok(())
    }

    /// Gives back the memory of the value planes' segments from `first` on,
    /// every column's, through `memory`, which retained them. The engine
    /// allocates none of their units and no device work accesses them any
    /// more.
    ///
    /// # Errors
    ///
    /// Fails for planes that are not split, or when `memory` fails.
    pub(crate) fn release_segments(
        &self,
        first: u32,
        memory: &mut dyn ExclusiveMemory,
    ) -> Result<()> {
        let (segmenting, device) = self.segmented()?;
        let from = segmenting.released_from.load(Ordering::Acquire);
        if first >= from {
            return Ok(());
        }
        if segmenting.segments[first as usize..from as usize]
            .iter()
            .any(|(_, inherited)| inherited.load(Ordering::Acquire))
        {
            return Err(Error::Invalid(
                "the predecessor world's segments are not given back by its successor".into(),
            ));
        }
        let peers = Storage::peers(device)?;
        let indices = segmenting.parts_of(first..from);
        for (name, storage) in ["prefix_cache.key.values", "prefix_cache.value.values"]
            .into_iter()
            .zip(&self.values)
        {
            memory.release_segments(
                &SegmentedSpec {
                    name,
                    device,
                    parts: &segmenting.parts,
                    access: &peers,
                },
                storage,
                &indices,
            )?;
        }
        segmenting.released_from.store(first, Ordering::Release);
        tracing::info!(
            first,
            segments = from - first,
            units = segmenting.first_unit(from) - segmenting.first_unit(first),
            "gave back cache segments"
        );
        Ok(())
    }

    /// Creates the segments given back again at their addresses, the first
    /// `limit` of them or all, through `memory`, and resets their units on
    /// `stream`, which is synchronized before returning: values and flags
    /// zero, scales one.
    ///
    /// # Errors
    ///
    /// Fails for planes that are not split, or when `memory` or the stream
    /// fails.
    pub(crate) fn restore_segments(
        &self,
        memory: &mut dyn ExclusiveMemory,
        stream: &Stream,
        limit: Option<u32>,
    ) -> Result<()> {
        let (segmenting, device) = self.segmented()?;
        let from = segmenting.released_from.load(Ordering::Acquire);
        let count = limit.map_or(segmenting.count(), |limit| {
            from.saturating_add(limit).min(segmenting.count())
        });
        if from == count {
            return Ok(());
        }
        let peers = Storage::peers(device)?;
        let indices = segmenting.parts_of(from..count);
        for (name, storage) in ["prefix_cache.key.values", "prefix_cache.value.values"]
            .into_iter()
            .zip(&self.values)
        {
            memory.restore_segments(
                &SegmentedSpec {
                    name,
                    device,
                    parts: &segmenting.parts,
                    access: &peers,
                },
                storage,
                &indices,
            )?;
        }
        let first_unit = segmenting.first_unit(from);
        let units = segmenting.first_unit(count) - first_unit;
        self.reset_units(first_unit, units, true, Some(stream))?;
        stream.synchronize()?;
        segmenting.released_from.store(count, Ordering::Release);
        tracing::info!(
            first = from,
            segments = count - from,
            "created the cache segments again"
        );
        Ok(())
    }

    fn segmented(&self) -> Result<(&Segmenting, DeviceId)> {
        match (&self.segmenting, self.device) {
            (Some(segmenting), Some(device)) => Ok((segmenting, device)),
            _ => Err(Error::Invalid(format!(
                "the cache planes of {} units are not split into segments, which give \
                 memory back",
                self.num_units
            ))),
        }
    }

    /// Resets units `[first, first + count)` of every column with native
    /// fills issued to `stream`: values (when `values`) and flags zero,
    /// scales one. The values of segmented planes fill column by column: a
    /// fill's rows may not span a segment given back, which the driver
    /// refuses although no row touches it.
    fn reset_units(&self, first: u32, count: u32, values: bool, stream: impl Issue) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        self.require_unit(first + count - 1)?;
        let units = self.num_units as usize;
        let columns = self.plan.columns as usize;
        let plane = self.plan.plane_bytes as usize;
        let (first, count) = (first as usize, count as usize);
        // Units `[first, first + count)` of every column: one row per
        // column, a column's units apart.
        let rows = |address: u64, value: u32, width: usize, cell: usize| RowFill {
            address: address + (first * cell * width) as u64,
            pitch: units * cell * width,
            value,
            width,
            count: count * cell,
            height: columns,
        };
        for field in 0..2 {
            // SAFETY: the rows lie inside the field allocations owned by
            // this pool, whose units the caller has mapped; the caller
            // orders the fills.
            unsafe {
                if values && self.segmenting.is_some() {
                    let fill = rows(self.values[field].address(), 0, 1, plane);
                    for column in 0..columns {
                        stream.memset_2d(RowFill {
                            address: fill.address + (column * fill.pitch) as u64,
                            height: 1,
                            ..fill
                        })?;
                    }
                } else if values {
                    stream.memset_2d(rows(self.values[field].address(), 0, 1, plane))?;
                }
                stream.memset_2d(rows(self.initialized[field].address(), 0, 1, 1))?;
                if let Some(scales) = &self.scales {
                    stream.memset_2d(rows(scales[field].address(), UNIT_SCALE, 4, 1))?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn plan(&self) -> &Arc<CachePlan> {
        &self.plan
    }

    pub(crate) fn device(&self) -> Option<DeviceId> {
        self.device
    }

    pub(crate) fn num_units(&self) -> u32 {
        self.num_units
    }

    fn cell(&self, column: u32, unit: u32) -> u64 {
        u64::from(column) * u64::from(self.num_units) + u64::from(unit)
    }

    fn require_unit(&self, unit: u32) -> Result<()> {
        if unit >= self.num_units {
            return Err(Error::Invalid(format!(
                "cache unit {unit} is outside the pool of {} units",
                self.num_units
            )));
        }
        Ok(())
    }

    /// The pool's unit fields as the library's cache classes take them: the
    /// list of their names, then one tensor per field.
    pub(crate) fn field_arguments(&self) -> Result<Vec<Any>> {
        let names = strings(self.fields.iter().map(|(name, _)| name));
        let mut arguments = vec![Any::from(names)];
        for (_, view) in &self.fields {
            arguments.push(view.handle()?.to_any());
        }
        Ok(arguments)
    }

    /// The bound field tensors' views, keyed as the numerical library reads
    /// them (`key.values`, `key.scale`, `key.initialized`, ...).
    pub fn views(&self) -> Vec<(String, StorageView)> {
        self.fields.clone()
    }

    /// Each field's whole view, in storage order.
    fn field_views(&self) -> Result<Vec<(String, StorageView)>> {
        let columns = i64::from(self.plan.columns);
        let units = i64::from(self.num_units);
        let mut views = Vec::new();
        for (field, name) in FIELDS.iter().enumerate() {
            views.push((
                format!("{name}.values"),
                StorageView::contiguous(
                    Arc::clone(&self.values[field]) as Arc<dyn Backing>,
                    0,
                    DType::U8,
                    &[columns, units, self.plan.plane_bytes as i64],
                )?,
            ));
            if let Some(scales) = &self.scales {
                views.push((
                    format!("{name}.scale"),
                    StorageView::contiguous(
                        Arc::clone(&scales[field]) as Arc<dyn Backing>,
                        0,
                        DType::F32,
                        &[columns, units],
                    )?,
                ));
            }
            views.push((
                format!("{name}.initialized"),
                StorageView::contiguous(
                    Arc::clone(&self.initialized[field]) as Arc<dyn Backing>,
                    0,
                    DType::Bool,
                    &[columns, units],
                )?,
            ));
        }
        Ok(views)
    }

    /// Tokens `[offset, offset + count)` of every column of `unit`, as group
    /// `group` reads them: `[count, columns, heads, dim]` in the group's
    /// stored dtype. Column `c` holds the page's layer `row * columns + c`
    /// of the unit's page position `row`.
    pub fn page(
        &self,
        field: usize,
        group: usize,
        unit: u32,
        offset: u32,
        count: u32,
    ) -> Result<StorageView> {
        self.require_unit(unit)?;
        let plan = &self.plan.groups[group];
        if offset + count > plan.block_size {
            return Err(Error::Invalid("a page view exceeds its page".into()));
        }
        let element = plan.dtype.bytes() as u64;
        let row = i64::from(plan.num_kv_heads) * i64::from(plan.head_dim);
        let base = u64::from(unit) * self.plan.plane_bytes + u64::from(offset) * plan.row_bytes();
        StorageView::strided(
            Arc::clone(&self.values[field]) as Arc<dyn Backing>,
            base as usize,
            plan.dtype,
            &[
                i64::from(count),
                i64::from(self.plan.columns),
                i64::from(plan.num_kv_heads),
                i64::from(plan.head_dim),
            ],
            &[
                row,
                (u64::from(self.num_units) * self.plan.plane_bytes / element) as i64,
                i64::from(plan.head_dim),
                1,
            ],
        )
    }

    /// Tokens `[offset, offset + count)` of one layer: column `column` of
    /// `unit`, as a contiguous `[count, 1, heads, dim]` group-dtype view
    /// matching one layer of a transfer's `[tokens, layers, heads, dim]`.
    pub(crate) fn layer_tokens(
        &self,
        field: usize,
        group: usize,
        column: u32,
        unit: u32,
        offset: u32,
        count: u32,
    ) -> Result<StorageView> {
        self.require_unit(unit)?;
        let plan = &self.plan.groups[group];
        if offset + count > plan.block_size || column >= self.plan.columns {
            return Err(Error::Invalid("a layer view exceeds its plane".into()));
        }
        let row = i64::from(plan.num_kv_heads) * i64::from(plan.head_dim);
        let base =
            self.cell(column, unit) * self.plan.plane_bytes + u64::from(offset) * plan.row_bytes();
        StorageView::strided(
            Arc::clone(&self.values[field]) as Arc<dyn Backing>,
            base as usize,
            plan.dtype,
            &[
                i64::from(count),
                1,
                i64::from(plan.num_kv_heads),
                i64::from(plan.head_dim),
            ],
            &[row, row, i64::from(plan.head_dim), 1],
        )
    }

    /// One unit's FP8 scale of `field` in `column`, as `[1, 1, 1, 1]`.
    pub(crate) fn scale(&self, field: usize, column: u32, unit: u32) -> Result<StorageView> {
        self.require_unit(unit)?;
        let scales = self
            .scales
            .as_ref()
            .ok_or_else(|| Error::Invalid("unquantized cache planes carry no scales".into()))?;
        StorageView::contiguous(
            Arc::clone(&scales[field]) as Arc<dyn Backing>,
            (self.cell(column, unit) * 4) as usize,
            DType::F32,
            &[1, 1, 1, 1],
        )
    }

    /// The address of one unit's plane of `field` in `column`: the
    /// `plane_bytes` of one layer of the unit's page.
    pub(crate) fn plane_address(&self, field: usize, column: u32, unit: u32) -> Result<u64> {
        self.require_unit(unit)?;
        Ok(self.values[field].address() + self.cell(column, unit) * self.plan.plane_bytes)
    }

    /// The address of one unit's FP8 scale of `field` in `column`.
    pub(crate) fn scale_address(&self, field: usize, column: u32, unit: u32) -> Result<u64> {
        Ok(self.scale(field, column, unit)?.address())
    }

    /// Reset units `1..=count` (contiguous startup scratch) with native
    /// fills issued to `stream`: values and flags zero, scales one.
    pub fn zero_leading(&self, count: u32, stream: impl Issue) -> Result<()> {
        self.reset_units(1, count, true, stream)
    }
}

/// One stream's binding of the scattered unit resets: a Rust-owned column of
/// unit ids and the Python operations that read its prefix.
///
/// A column is reused only in its stream's order: an upload for the next
/// operation is ordered after the previous operation's kernels.
pub struct UnitColumn {
    units: StorageView,
    stream: Option<usize>,
    /// The library's `UnitResets` over the column and the pool's fields.
    resets: Handle,
    zero: Function,
    recycle: Function,
    mark_initialized: Function,
}

/// The library class of the scattered unit resets.
const UNIT_RESETS: &str = "uniserve.prefix_cache.units.UnitResets";

impl UnitColumn {
    /// Device bytes of a column over a pool of `num_units` units: one int64
    /// unit index each.
    pub(crate) fn bytes(num_units: u32) -> u64 {
        u64::from(num_units) * DType::I64.bytes() as u64
    }

    /// Allocate a column for `planes` and construct the library's
    /// `UnitResets` over it on `stream` (`None` for a CPU pool).
    pub(crate) fn bind(planes: &CachePlanes, stream: Option<&Stream>) -> Result<Self> {
        let count = i64::from(planes.num_units);
        let bytes = Self::bytes(planes.num_units) as usize;
        // A column is rebuilt with the stream that binds it, so it is the
        // worker's own memory.
        let memory = match planes.device {
            Some(device) => Storage::device(device, bytes)?,
            None => Storage::host(bytes),
        };
        let units = StorageView::contiguous(memory as Arc<dyn Backing>, 0, DType::I64, &[count])?;

        let mut arguments = vec![
            units.handle()?.to_any(),
            Any::from(stream.map_or(0, |stream| stream.raw() as i64)),
        ];
        arguments.extend(planes.field_arguments()?);
        let resets = Handle::new(Interpreter::import(UNIT_RESETS)?.call_values(&arguments)?)?;
        let method = |name: &str| Interpreter::import(&format!("{UNIT_RESETS}.{name}"));
        Ok(Self {
            units,
            stream: stream.map(Stream::raw),
            resets,
            zero: method("zero")?,
            recycle: method("recycle")?,
            mark_initialized: method("mark_initialized")?,
        })
    }

    /// Write `units` into the column prefix in `stream` order.
    fn upload(&self, planes: &CachePlanes, units: &[u32], stream: impl Issue) -> Result<i64> {
        if stream.stream().map(Stream::raw) != self.stream {
            return Err(Error::Invariant(
                "a unit column is used on the stream it was bound to".into(),
            ));
        }
        if units.len() > planes.num_units as usize {
            return Err(Error::Invalid(
                "more cache units than the pool holds".into(),
            ));
        }
        let ids = units
            .iter()
            .map(|&unit| planes.require_unit(unit).map(|()| i64::from(unit)))
            .collect::<Result<Vec<_>>>()?;
        let bytes = ids.len() * size_of::<i64>();
        let mut batch = CopyBatch::new(SourceOrder::DuringCall);
        batch.push(self.units.address(), ids.as_ptr() as u64, bytes);
        match self.units.device() {
            // SAFETY: the ids are read during the call; the column write is
            // ordered on the stream after the previous operation that read
            // it.
            MemoryKind::Cuda(_) if stream.stream().is_some() => unsafe { stream.copy(&batch) }?,
            // SAFETY: a host column is written in its issuer's order, after
            // the previous operation that read it.
            _ => unsafe { stream.host().copy(&batch) }?,
        }
        Ok(ids.len() as i64)
    }

    /// Upload `units` and run `function` over the column's prefix, in
    /// `stream` order.
    fn apply(
        &self,
        function: &Function,
        planes: &CachePlanes,
        units: &[u32],
        stream: impl Issue,
    ) -> Result<()> {
        if units.is_empty() {
            return Ok(());
        }
        let count = self.upload(planes, units, stream)?;
        stream.call(DeferredCall::new(
            function.clone(),
            vec![self.resets.to_any(), Any::from(count)],
        ))
    }

    /// Reset `units` completely: values zero, flags clear, scales one.
    pub(crate) fn zero(
        &self,
        planes: &CachePlanes,
        units: &[u32],
        stream: impl Issue,
    ) -> Result<()> {
        self.apply(&self.zero, planes, units, stream)
    }

    /// Prepare units that held earlier tokens for a new owner. Readers never
    /// weigh unwritten positions, so stale finite values may stay: a
    /// single-dtype unquantized pool needs no device work, a quantized one
    /// resets flags and scales, and a mixed-dtype pool is zeroed because
    /// stale bytes of one type need not decode as finite values of another.
    pub(crate) fn recycle(
        &self,
        planes: &CachePlanes,
        units: &[u32],
        stream: impl Issue,
    ) -> Result<()> {
        if planes.plan.mixed_types() {
            self.zero(planes, units, stream)
        } else if planes.plan.quantized {
            self.apply(&self.recycle, planes, units, stream)
        } else {
            Ok(())
        }
    }

    /// Commit transferred values and scales of whole units: set both
    /// fields' initialization flags in every column.
    pub(crate) fn mark_initialized(
        &self,
        planes: &CachePlanes,
        units: &[u32],
        stream: impl Issue,
    ) -> Result<()> {
        self.apply(&self.mark_initialized, planes, units, stream)
    }
}
