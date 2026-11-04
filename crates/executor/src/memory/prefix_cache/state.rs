//! The state group's block copy: a page's checkpoint bytes moved between the pool and a
//! contiguous snapshot, in the two directions §8 names (`cache_block_snapshot` and
//! `cache_block_scatter` of `ops::block_copy`).
//!
//! Both programs name their blocks with a device `i64` vector, so no host read stands between the
//! snapshot and the scatter and a captured graph replays the pair. The index vector is filled on
//! the host - the page's units are the scheduler's choice - and copied to the device on the same
//! stream, which is a write before the launch and not a read between two launches.
//!
//! The pool stores a layer's checkpoint as one plane per (column, unit) cell, cell-major:
//! `values[field]` holds `columns * num_units` planes of `plane_bytes` bytes each, so a plane is
//! addressed by its index alone and the copy is a block copy of raw bytes whatever the checkpoint
//! encodes.

use ops::block_copy::{
    BlockShape, COPY_BLOCK, CacheBlockScatter, CacheBlockSnapshot, ScatterArgs, SnapshotArgs,
};
use ops::{DType, Shape, Strides, TensorView};
use platform::{Buffer, PoolKind, Stream};

use crate::Error;
use crate::Result;
use crate::arena::Pools;
use crate::memory::prefix_cache::plan::{CacheGroupPlan, Checkpoint};

/// The two programs a state copy launches, in the order it launches them.
pub type SnapshotOp = CacheBlockSnapshot;
/// The scatter that undoes a [`SnapshotOp`].
pub type ScatterOp = CacheBlockScatter;

/// A contiguous device buffer of `rows` blocks of `width` bytes, with the device index vector the
/// two programs select blocks by.
///
/// The caller allocates one per state group from [`Pools`], before the first batch, and hands the
/// same snapshot to every copy: nothing here allocates during a step.
pub struct Snapshot {
    /// `rows * width` bytes of raw checkpoint storage.
    data: Buffer,
    /// `rows` `i64` device indices: the block each snapshot row takes, or writes.
    blocks: Buffer,
    /// The host copy of the index vector, which the device vector is filled from.
    staging: Vec<i64>,
    rows: u32,
    width: u64,
}

impl Snapshot {
    /// Takes a snapshot of `rows` blocks of `width` bytes from `pools`' scratch storage.
    ///
    /// # Errors
    ///
    /// The pool's error when either buffer cannot be mapped, and [`Error::Invalid`] when the
    /// extent is zero.
    pub fn allocate(pools: &mut Pools, rows: u32, width: u64) -> Result<Self> {
        if rows == 0 || width == 0 {
            return Err(Error::Invalid(
                "a state snapshot holds at least one block of at least one byte".to_owned(),
            ));
        }
        let bytes = u64::from(rows) * width;
        Ok(Self {
            data: pools.allocate(PoolKind::Scratch, bytes)?,
            blocks: pools.allocate(PoolKind::Scratch, u64::from(rows) * 8)?,
            staging: vec![0; rows as usize],
            rows,
            width,
        })
    }

    /// The blocks the snapshot holds: the most one copy may name.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Bytes one block of the snapshot holds.
    pub fn width(&self) -> u64 {
        self.width
    }

    /// Bytes the snapshot's data occupies, which is what a cache export of a state page states.
    pub fn bytes(&self) -> u64 {
        u64::from(self.rows) * self.width
    }

    /// Fills the device index vector with `blocks` on `stream`: row `k` of the next copy reads or
    /// writes block `blocks[k]`. A negative index reads block zero and writes nothing, which is
    /// the programs' own rule for an inactive pair's placeholder.
    ///
    /// The vector stays in the snapshot, so a caller fills it once per page it copies.
    pub fn write_blocks(&mut self, blocks: &[i64], stream: &Stream) -> Result<()> {
        if blocks.len() > self.staging.len() {
            return Err(Error::Invalid(format!(
                "a snapshot of {} rows cannot name {} blocks",
                self.rows,
                blocks.len()
            )));
        }
        for (at, block) in blocks.iter().enumerate() {
            self.staging[at] = *block;
        }
        let bytes = blocks.len() * 8;
        // SAFETY: the source is this snapshot's staging vector, which outlives the copy because
        // the snapshot owns it and the caller keeps the snapshot until the stream is done; the
        // destination is the index vector's allocation of `rows * 8` bytes.
        unsafe {
            platform::memcpy_async(
                self.blocks.address(),
                self.staging.as_ptr() as u64,
                bytes,
                stream,
            )
        }
        .map_err(Into::into)
    }

    /// The rows `[offset, offset + count)` of the snapshot as the copy's operand: `[count,
    /// width]`, contiguous.
    pub(crate) fn view(&self, offset: u32, count: u32) -> Result<TensorView> {
        if offset.checked_add(count).is_none_or(|end| end > self.rows) {
            return Err(Error::Invalid(format!(
                "a copy of {count} blocks exceeds the snapshot's {} rows",
                self.rows
            )));
        }
        Ok(TensorView::new(
            ops::DevicePtr::new(self.data.address() + u64::from(offset) * self.width),
            DType::U8,
            Shape::new(&[u64::from(count), self.width]),
        ))
    }

    /// The index vector's rows `[offset, offset + count)` as the copy's operand.
    pub(crate) fn blocks_view(&self, offset: u32, count: u32) -> Result<TensorView> {
        Ok(TensorView::new(
            ops::DevicePtr::new(self.blocks.address() + u64::from(offset) * 8),
            DType::I64,
            Shape::new(&[u64::from(count)]),
        ))
    }
}

/// One layer's checkpoint planes, as the block copy reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatePlanes {
    /// Bytes one plane of the checkpoint occupies: the width of one block.
    pub width: u64,
    /// Planes of the layer's checkpoint, in plane order.
    pub planes: u32,
}

/// The descriptor of one block copy: `count` blocks of `width` bytes of raw storage.
pub(crate) fn descriptor(count: u32, width: u64) -> BlockShape {
    let shape = Shape::new(&[u64::from(count), width]);
    BlockShape {
        dtype: DType::U8,
        shape,
        strides: Strides::row_major(shape),
        index_stride: 1,
        width,
        block: COPY_BLOCK,
    }
}

/// The blocks one page's checkpoint occupies, per storage field: field `f`'s cells are the
/// planes of that field, in plane order, and a run of planes in one field is one launch.
///
/// The pool stores `columns * num_units` planes per field, cell-major, so a plane is addressed by
/// `column * num_units + unit` alone. A plane index below `first_plane` or beyond the checkpoint
/// belongs to another layer and is refused.
pub(crate) fn page_cells(
    group: &CacheGroupPlan,
    columns: u32,
    num_units: u32,
    checkpoint: &Checkpoint,
    units: &[u32],
) -> Result<Vec<(usize, Vec<i64>)>> {
    let half = group.units_per_page * columns;
    if half == 0 {
        return Err(Error::Invariant(format!(
            "a state page of {} layers holds no unit",
            group.layers.len()
        )));
    }
    let mut by_field: Vec<(usize, Vec<i64>)> = Vec::new();
    for index in 0..checkpoint.planes {
        let position = checkpoint.first_plane + index;
        let field = (position / half) as usize;
        let slot = position % half;
        let unit = *units.get((slot / columns) as usize).ok_or_else(|| {
            Error::Invalid(format!(
                "a page of {} units cannot hold plane {position}",
                units.len()
            ))
        })?;
        let cell = i64::from(slot % columns) * i64::from(num_units) + i64::from(unit);
        match by_field.last_mut() {
            Some((held, cells)) if *held == field => cells.push(cell),
            _ => by_field.push((field, vec![cell])),
        }
    }
    Ok(by_field)
}

/// The two copies' arguments, in one place so a caller states them once.
pub(crate) fn snapshot_args(
    field: TensorView,
    blocks: TensorView,
    snapshot: TensorView,
) -> SnapshotArgs {
    SnapshotArgs {
        field,
        blocks,
        snapshot,
    }
}

/// See [`snapshot_args`].
pub(crate) fn scatter_args(
    field: TensorView,
    blocks: TensorView,
    snapshot: TensorView,
) -> ScatterArgs {
    ScatterArgs {
        field,
        blocks,
        snapshot,
    }
}
