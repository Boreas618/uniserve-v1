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

use std::sync::Arc;

use cuda::{CopyBatch, DeviceId, SourceOrder, Stream};
use tvm_ffi::Any;

use super::plan::CachePlan;
use crate::ffi::{
    Backing, DType, DeferredCall, Function, Handle, Interpreter, MemoryKind, strings,
};
use crate::memory::allocation::{Storage, StorageView};
use crate::memory::retained::{ExclusiveMemory, ExclusiveSpec};
use crate::stream::{Issue, RowFill};
use crate::{Error, Result};

/// The two K/V fields, in storage order.
pub(crate) const FIELDS: [&str; 2] = ["key", "value"];

/// The bit pattern of `1.0f32`, the neutral FP8 scale.
const UNIT_SCALE: u32 = 0x3F80_0000;

/// The pool's storage on one device (or host memory for CPU lanes).
pub struct CachePlanes {
    plan: Arc<CachePlan>,
    device: Option<DeviceId>,
    num_units: u32,
    values: [Arc<Storage>; 2],
    scales: Option<[Arc<Storage>; 2]>,
    initialized: [Arc<Storage>; 2],
    /// Each field's whole view, keyed as the numerical library reads it.
    /// Every binding passes these views, so each field is wrapped as a
    /// torch tensor once.
    fields: Vec<(String, StorageView)>,
}

impl CachePlanes {
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
    /// A dense cache export aliases the live pages, so the values are
    /// exported where they lie, whatever the pool's size, and every local
    /// peer device may address them, because readers on those devices in
    /// this process copy from the pages directly.
    pub fn allocate(
        plan: Arc<CachePlan>,
        device: Option<DeviceId>,
        num_units: u32,
        stream: Option<&Stream>,
        memory: &mut dyn ExclusiveMemory,
    ) -> Result<Self> {
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
        let (values, initialized, scales) = match device {
            Some(device) => {
                let peers = Storage::peers(device)?;
                let spec = |name, bytes| ExclusiveSpec {
                    name,
                    device,
                    bytes,
                    access: &peers,
                };
                let mut specs = vec![
                    spec("prefix_cache.key.values", plane_bytes),
                    spec("prefix_cache.value.values", plane_bytes),
                    spec("prefix_cache.key.initialized", cells),
                    spec("prefix_cache.value.initialized", cells),
                ];
                if plan.quantized {
                    specs.push(spec("prefix_cache.key.scale", cells * 4));
                    specs.push(spec("prefix_cache.value.scale", cells * 4));
                }
                let taken = memory.exclusive(&specs)?;
                for (index, created) in created.iter_mut().enumerate() {
                    *created = taken
                        .iter()
                        .skip(2 * index)
                        .take(2)
                        .any(|storage| storage.created);
                }
                let mut taken = taken.into_iter();
                let mut next = || {
                    taken
                        .next()
                        .map(|taken| taken.storage)
                        .ok_or_else(|| Error::Invariant("an exclusive storage is missing".into()))
                };
                let values = [next()?, next()?];
                let initialized = [next()?, next()?];
                let scales = if plan.quantized {
                    Some([next()?, next()?])
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
        if let Some(stream) = stream {
            stream.synchronize()?;
        }
        Ok(planes)
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
        if count == 0 {
            return Ok(());
        }
        self.require_unit(count)?;
        let units = self.num_units as usize;
        let columns = self.plan.columns as usize;
        let plane = self.plan.plane_bytes as usize;
        let count = count as usize;
        // Units `[1, count]` of every column: one row per column, a
        // column's units apart.
        let leading = |address: u64, value: u32, width: usize, cell: usize| RowFill {
            address: address + (cell * width) as u64,
            pitch: units * cell * width,
            value,
            width,
            count: count * cell,
            height: columns,
        };
        for field in 0..2 {
            // SAFETY: rows `[1, count]` of each column lie inside the field
            // allocations owned by this pool; the caller orders the fills.
            unsafe {
                stream.memset_2d(leading(self.values[field].address(), 0, 1, plane))?;
                stream.memset_2d(leading(self.initialized[field].address(), 0, 1, 1))?;
                if let Some(scales) = &self.scales {
                    stream.memset_2d(leading(scales[field].address(), UNIT_SCALE, 4, 1))?;
                }
            }
        }
        Ok(())
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
        // A column is rebuilt with its lane, so it is the worker's own memory.
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
