//! The KV unit pool's device storage and its unit-level resets.
//!
//! Each field (key, value) owns separate allocations in the layout the
//! numerical cache library reads (`uniserve_models.cache.units`):
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

use tvm_ffi::AnyView;
use uniserve_cuda::{CopyBatch, SourceOrder, Stream, memset_2d_d8, memset_2d_d32};

use super::kv_plan::KVPlan;
use super::memory::{Memory, StorageView};
use crate::ffi::{Backing, DType, Device, Function, Functions, Tensor};
use crate::{Error, Result};

/// The two K/V fields, in storage order.
pub const FIELDS: [&str; 2] = ["key", "value"];

/// The bit pattern of `1.0f32`, the neutral FP8 scale.
const UNIT_SCALE: u32 = 0x3F80_0000;

/// The pool's storage on one device (or host memory for CPU lanes).
pub struct KVPlanes {
    plan: Arc<KVPlan>,
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

/// Fill `height` rows of `width` bytes, `pitch` bytes apart, with `value`.
///
/// # Safety
/// Every row lies in the memory of `stream`'s device (or host memory when
/// `stream` is `None`) and stays valid until the fill completes.
unsafe fn fill_d8(
    address: u64,
    pitch: usize,
    value: u8,
    width: usize,
    height: usize,
    stream: Option<&Stream>,
) -> Result<()> {
    match stream {
        // SAFETY: forwarded from the caller.
        Some(stream) => unsafe { memset_2d_d8(address, pitch, value, width, height, stream) }?,
        None => {
            for row in 0..height {
                // SAFETY: forwarded from the caller.
                unsafe {
                    std::ptr::write_bytes(
                        (address + (row * pitch) as u64) as *mut u8,
                        value,
                        width,
                    );
                }
            }
        }
    }
    Ok(())
}

/// Fill `height` rows of `width` 32-bit words, `pitch` bytes apart.
///
/// # Safety
/// As for [`fill_d8`], with 4-byte aligned rows.
unsafe fn fill_d32(
    address: u64,
    pitch: usize,
    value: u32,
    width: usize,
    height: usize,
    stream: Option<&Stream>,
) -> Result<()> {
    match stream {
        // SAFETY: forwarded from the caller.
        Some(stream) => unsafe { memset_2d_d32(address, pitch, value, width, height, stream) }?,
        None => {
            for row in 0..height {
                let start = (address + (row * pitch) as u64) as *mut u32;
                for word in 0..width {
                    // SAFETY: forwarded from the caller.
                    unsafe { start.add(word).write(value) };
                }
            }
        }
    }
    Ok(())
}

impl KVPlanes {
    /// Device bytes of a pool of `num_units` units.
    pub fn bytes(plan: &KVPlan, num_units: u32) -> u64 {
        plan.unit_bytes * u64::from(num_units)
    }

    /// Allocate and reset a pool of `num_units` units (the sentinel
    /// included): values and flags zero, scales one. On a device the resets
    /// run on `stream`, which is synchronized before returning.
    pub fn allocate(
        plan: Arc<KVPlan>,
        device: Option<i32>,
        num_units: u32,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        if num_units < 2 {
            return Err(Error::Invalid(
                "a KV unit pool needs the sentinel and one usable unit".into(),
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
            allocate(device, plane_bytes)?,
            allocate(device, plane_bytes)?,
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

        for field in 0..2 {
            // SAFETY: each allocation spans exactly the filled bytes and is
            // owned by `planes`; the stream is drained below.
            unsafe {
                fill_d8(
                    planes.values[field].address(),
                    plane_bytes,
                    0,
                    plane_bytes,
                    1,
                    stream,
                )?;
                fill_d8(
                    planes.initialized[field].address(),
                    cells,
                    0,
                    cells,
                    1,
                    stream,
                )?;
                if let Some(scales) = &planes.scales {
                    fill_d32(
                        scales[field].address(),
                        cells * 4,
                        UNIT_SCALE,
                        cells,
                        1,
                        stream,
                    )?;
                }
            }
        }
        if let Some(stream) = stream {
            stream.synchronize()?;
        }
        Ok(planes)
    }

    pub fn plan(&self) -> &Arc<KVPlan> {
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
                "KV unit {unit} is outside the pool of {} units",
                self.num_units
            )));
        }
        Ok(())
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
            .ok_or_else(|| Error::Invalid("unquantized KV planes carry no scales".into()))?;
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
    /// fills: values and flags zero, scales one.
    pub fn zero_leading(&self, count: u32, stream: Option<&Stream>) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        self.require_unit(count)?;
        let units = self.num_units as usize;
        let columns = self.plan.columns as usize;
        let plane = self.plan.plane_bytes as usize;
        for field in 0..2 {
            // SAFETY: rows `[1, count]` of each column lie inside the field
            // allocations owned by this pool; the caller orders the fills.
            unsafe {
                fill_d8(
                    self.values[field].address() + plane as u64,
                    units * plane,
                    0,
                    count as usize * plane,
                    columns,
                    stream,
                )?;
                fill_d8(
                    self.initialized[field].address() + 1,
                    units,
                    0,
                    count as usize,
                    columns,
                    stream,
                )?;
                if let Some(scales) = &self.scales {
                    fill_d32(
                        scales[field].address() + 4,
                        units * 4,
                        UNIT_SCALE,
                        count as usize,
                        columns,
                        stream,
                    )?;
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
/// them (prefill graph warmup writes real KV pages).
pub struct UnitSnapshot {
    units: Vec<u32>,
    memory: Arc<Memory>,
}

impl UnitSnapshot {
    /// Copy every byte of `units` into a new snapshot on `stream`. The
    /// snapshot is consistent once the stream reaches the copy.
    pub fn save(planes: &KVPlanes, units: &[u32], stream: &Stream) -> Result<Self> {
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
    pub fn restore(self, planes: &KVPlanes, stream: &Stream) -> Result<()> {
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
    zero: Function,
    recycle: Function,
    mark_initialized: Function,
}

impl UnitColumn {
    /// Allocate a column for `planes` and bind `cache.bind_units` to it on
    /// `stream` (`None` for a CPU pool).
    pub fn bind(planes: &KVPlanes, bind_units: &Function, stream: Option<&Stream>) -> Result<Self> {
        let count = i64::from(planes.num_units);
        let memory = allocate(planes.device, planes.num_units as usize * 8)?;
        let units = StorageView::contiguous(memory as Arc<dyn Backing>, 0, DType::I64, &[count])?;

        let cache = tensor_map(&planes.views()?)?;
        let tensor = units.to_tensor()?;
        let handle = stream.map_or(0, |stream| stream.raw() as i64);
        let functions = Functions::new(
            bind_units.name(),
            bind_units.call(&[
                AnyView::from(&cache),
                AnyView::from(&tensor),
                AnyView::from(&handle),
            ])?,
        )?;
        Ok(Self {
            units,
            stream: stream.map(Stream::raw),
            zero: functions.get("zero")?,
            recycle: functions.get("recycle")?,
            mark_initialized: functions.get("mark_initialized")?,
        })
    }

    /// Write `units` into the column prefix in `stream` order.
    fn upload(&self, planes: &KVPlanes, units: &[u32], stream: Option<&Stream>) -> Result<i64> {
        if stream.map(Stream::raw) != self.stream {
            return Err(Error::Invariant(
                "a unit column is used on the stream it was bound to".into(),
            ));
        }
        if units.len() > planes.num_units as usize {
            return Err(Error::Invalid("more KV units than the pool holds".into()));
        }
        let ids = units
            .iter()
            .map(|&unit| planes.require_unit(unit).map(|()| i64::from(unit)))
            .collect::<Result<Vec<_>>>()?;
        let bytes = ids.len() * size_of::<i64>();
        match (self.units.device(), stream) {
            (Device::Cuda(_), Some(stream)) => {
                let mut batch = CopyBatch::new(SourceOrder::DuringCall);
                batch.push(self.units.address(), ids.as_ptr() as u64, bytes);
                // SAFETY: the driver reads the pageable ids during the call;
                // the column write is ordered on the stream after the
                // previous operation that read it.
                unsafe { batch.submit(stream) }?;
            }
            _ => {
                // SAFETY: a host column is written only by its owning thread,
                // after the previous synchronous operation returned.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ids.as_ptr(),
                        self.units.address() as *mut i64,
                        ids.len(),
                    );
                }
            }
        }
        Ok(ids.len() as i64)
    }

    fn apply(
        &self,
        function: &Function,
        planes: &KVPlanes,
        units: &[u32],
        stream: Option<&Stream>,
    ) -> Result<()> {
        if units.is_empty() {
            return Ok(());
        }
        let count = self.upload(planes, units, stream)?;
        function.invoke(&[AnyView::from(&count)])
    }

    /// Reset `units` completely: values zero, flags clear, scales one.
    pub fn zero(&self, planes: &KVPlanes, units: &[u32], stream: Option<&Stream>) -> Result<()> {
        self.apply(&self.zero, planes, units, stream)
    }

    /// Prepare units that held earlier tokens for a new owner. Readers never
    /// weigh unwritten positions, so stale finite values may stay: a
    /// single-dtype unquantized pool needs no device work, a quantized one
    /// resets flags and scales, and a mixed-dtype pool is zeroed because
    /// stale bytes of one type need not decode as finite values of another.
    pub fn recycle(&self, planes: &KVPlanes, units: &[u32], stream: Option<&Stream>) -> Result<()> {
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
        planes: &KVPlanes,
        units: &[u32],
        stream: Option<&Stream>,
    ) -> Result<()> {
        self.apply(&self.mark_initialized, planes, units, stream)
    }
}

/// Convert named views into the `Map<String, Tensor>` bindings receive.
/// Tensors are created on the calling thread, which owns them.
pub fn tensor_map(
    views: &[(String, StorageView)],
) -> Result<tvm_ffi::Map<tvm_ffi::String, Tensor>> {
    views
        .iter()
        .map(|(name, view)| Ok((tvm_ffi::String::from(name.as_str()), view.to_tensor()?)))
        .collect()
}
