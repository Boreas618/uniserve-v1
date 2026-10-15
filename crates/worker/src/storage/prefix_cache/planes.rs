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

use tvm_ffi::Any;
use uniserve_cuda::{CopyBatch, SourceOrder, Stream};

use super::plan::CachePlan;
use crate::ffi::{Backing, DType, DeferredCall, Device, Function, Handle, strings};
use crate::model::interface::Interface;
use crate::storage::memory::{Memory, StorageView};
use crate::stream::{Issue, RowFill};
use crate::{Error, Result};

/// The two K/V fields, in storage order.
pub const FIELDS: [&str; 2] = ["key", "value"];

/// The bit pattern of `1.0f32`, the neutral FP8 scale.
const UNIT_SCALE: u32 = 0x3F80_0000;

/// The pool's storage on one device (or host memory for CPU lanes).
pub struct CachePlanes {
    plan: Arc<CachePlan>,
    device: Option<i32>,
    num_units: u32,
    values: [Arc<Memory>; 2],
    scales: Option<[Arc<Memory>; 2]>,
    initialized: [Arc<Memory>; 2],
}

fn allocate(device: Option<i32>, bytes: usize) -> Result<Arc<Memory>> {
    match device {
        Some(device) => Memory::device(device, bytes),
        None => Ok(Memory::host(bytes)),
    }
}

/// Storage of one field's values. A dense cache export aliases the live pages,
/// so device planes are exportable allocations: a CUDA VMM export names the
/// pages where they lie, whatever the pool's size. Every local peer device
/// may address them, because readers on those devices in this process copy
/// from the pages directly.
fn allocate_values(device: Option<i32>, bytes: usize) -> Result<Arc<Memory>> {
    match device {
        Some(device) => Memory::exportable(device, bytes, &Memory::peers(device)?),
        None => Ok(Memory::host(bytes)),
    }
}

impl CachePlanes {
    /// Device bytes of a pool of `num_units` units.
    pub fn bytes(plan: &CachePlan, num_units: u32) -> u64 {
        plan.unit_bytes * u64::from(num_units)
    }

    /// Allocate and reset a pool of `num_units` units (the sentinel
    /// included): values and flags zero, scales one. On a device the resets
    /// run on `stream`, which is synchronized before returning.
    pub fn allocate(
        plan: Arc<CachePlan>,
        device: Option<i32>,
        num_units: u32,
        stream: Option<&Stream>,
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
        let values = [
            allocate_values(device, plane_bytes)?,
            allocate_values(device, plane_bytes)?,
        ];
        let initialized = [allocate(device, cells)?, allocate(device, cells)?];
        let scales = if plan.quantized {
            Some([allocate(device, cells * 4)?, allocate(device, cells * 4)?])
        } else {
            None
        };
        let planes = Self {
            plan,
            device,
            num_units,
            values,
            scales,
            initialized,
        };

        // One row spanning each allocation.
        let whole = |address: u64, value: u32, width: usize, count: usize| RowFill {
            address,
            pitch: count * width,
            value,
            width,
            count,
            height: 1,
        };
        for field in 0..2 {
            // SAFETY: each allocation spans exactly the filled bytes and is
            // owned by `planes`; the stream is drained below.
            unsafe {
                stream.memset_2d(whole(planes.values[field].address(), 0, 1, plane_bytes))?;
                stream.memset_2d(whole(planes.initialized[field].address(), 0, 1, cells))?;
                if let Some(scales) = &planes.scales {
                    stream.memset_2d(whole(scales[field].address(), UNIT_SCALE, 4, cells))?;
                }
            }
        }
        if let Some(stream) = stream {
            stream.synchronize()?;
        }
        Ok(planes)
    }

    pub fn plan(&self) -> &Arc<CachePlan> {
        &self.plan
    }

    pub fn device(&self) -> Option<i32> {
        self.device
    }

    pub fn num_units(&self) -> u32 {
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
    pub fn field_arguments(&self) -> Result<Vec<Any>> {
        let fields = self.views()?;
        let mut arguments = vec![Any::from(strings(fields.iter().map(|(name, _)| name)))];
        for (_, view) in &fields {
            arguments.push(Any::from(view.to_tensor()?));
        }
        Ok(arguments)
    }

    /// The bound field tensors' views, keyed as the numerical library reads
    /// them (`key.values`, `key.scale`, `key.initialized`, ...).
    pub fn views(&self) -> Result<Vec<(String, StorageView)>> {
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
        if offset + count > plan.page_tokens {
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
    pub fn layer_tokens(
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
        if offset + count > plan.page_tokens || column >= self.plan.columns {
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
    pub fn scale(&self, field: usize, column: u32, unit: u32) -> Result<StorageView> {
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

    /// The address of one unit's FP8 scale of `field` in `column`.
    pub fn scale_address(&self, field: usize, column: u32, unit: u32) -> Result<u64> {
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

    /// Plan copies of every byte of `units` (planes, flags, scales) between
    /// the pool and a snapshot buffer laid out unit-major. `save` copies
    /// pool to snapshot, otherwise snapshot to pool.
    fn snapshot_copies(&self, units: &[u32], snapshot: u64, save: bool) -> Result<CopyBatch> {
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        let mut cursor = snapshot;
        let mut push = |pool: u64, bytes: usize| {
            if save {
                batch.push(cursor, pool, bytes);
            } else {
                batch.push(pool, cursor, bytes);
            }
            cursor += bytes as u64;
        };
        for &unit in units {
            self.require_unit(unit)?;
            for field in 0..2 {
                for column in 0..self.plan.columns {
                    let cell = self.cell(column, unit);
                    push(
                        self.values[field].address() + cell * self.plan.plane_bytes,
                        self.plan.plane_bytes as usize,
                    );
                    push(self.initialized[field].address() + cell, 1);
                    if let Some(scales) = &self.scales {
                        push(scales[field].address() + cell * 4, 4);
                    }
                }
            }
        }
        Ok(batch)
    }
}

/// The saved bytes of a set of units, restored after a capture that wrote
/// them (prefill graph warmup writes real cache pages).
pub struct UnitSnapshot {
    units: Vec<u32>,
    memory: Arc<Memory>,
}

impl UnitSnapshot {
    /// Copy every byte of `units` into a new snapshot on `stream`. The
    /// snapshot is consistent once the stream reaches the copy.
    pub fn save(planes: &CachePlanes, units: &[u32], stream: &Stream) -> Result<Self> {
        let mut units = units.to_vec();
        units.sort_unstable();
        units.dedup();
        let device = planes
            .device
            .ok_or_else(|| Error::Invalid("unit snapshots serve device pools".into()))?;
        let memory = Memory::device(device, units.len().max(1) * planes.plan.unit_bytes as usize)?;
        let batch = planes.snapshot_copies(&units, memory.address(), true)?;
        // SAFETY: the snapshot and the pool stay alive until `restore`
        // drains the stream; distinct units never overlap.
        unsafe { batch.submit(stream) }?;
        Ok(Self { units, memory })
    }

    /// Restore the saved bytes on `stream` and drain it, releasing the
    /// snapshot.
    pub fn restore(self, planes: &CachePlanes, stream: &Stream) -> Result<()> {
        let batch = planes.snapshot_copies(&self.units, self.memory.address(), false)?;
        stream.synchronize()?;
        // SAFETY: as for `save`; the stream is drained before the snapshot drops.
        unsafe { batch.submit(stream) }?;
        stream.synchronize()?;
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
const UNIT_RESETS: &str = "uniserve.interface.cache.UnitResets";

impl UnitColumn {
    /// Allocate a column for `planes` and construct the library's
    /// `UnitResets` over it on `stream` (`None` for a CPU pool).
    pub fn bind(
        planes: &CachePlanes,
        interface: &Interface,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        let count = i64::from(planes.num_units);
        let memory = allocate(planes.device, planes.num_units as usize * 8)?;
        let units = StorageView::contiguous(memory as Arc<dyn Backing>, 0, DType::I64, &[count])?;

        let mut arguments = vec![
            Any::from(units.to_tensor()?),
            Any::from(stream.map_or(0, |stream| stream.raw() as i64)),
        ];
        arguments.extend(planes.field_arguments()?);
        let resets = Handle::new(interface.import(UNIT_RESETS)?.call_values(&arguments)?)?;
        let method = |name: &str| interface.import(&format!("{UNIT_RESETS}.{name}"));
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
            Device::Cuda(_) if stream.stream().is_some() => unsafe { stream.copy(&batch) }?,
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
    pub fn zero(&self, planes: &CachePlanes, units: &[u32], stream: impl Issue) -> Result<()> {
        self.apply(&self.zero, planes, units, stream)
    }

    /// Prepare units that held earlier tokens for a new owner. Readers never
    /// weigh unwritten positions, so stale finite values may stay: a
    /// single-dtype unquantized pool needs no device work, a quantized one
    /// resets flags and scales, and a mixed-dtype pool is zeroed because
    /// stale bytes of one type need not decode as finite values of another.
    pub fn recycle(&self, planes: &CachePlanes, units: &[u32], stream: impl Issue) -> Result<()> {
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
    pub fn mark_initialized(
        &self,
        planes: &CachePlanes,
        units: &[u32],
        stream: impl Issue,
    ) -> Result<()> {
        self.apply(&self.mark_initialized, planes, units, stream)
    }
}
