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

    /// Address of the snapshot's first block: the caller that fills it before a scatter, or reads
    /// it after a snapshot, copies through this.
    pub fn address(&self) -> u64 {
        self.data.address()
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

/// The blocks of every unit of `pages` pages starting at `first_page`, per storage field, in the
/// page's own order.
///
/// This is the layout of a state page in either direction: an export gathers these cells into
/// its rows, an import scatters the same rows back into them, and the block copy of one page is
/// the same list. `table` is the slot's block table of the state group, so the units are the
/// destination's own; `num_units` is the pool's, which the cell indices are strided by.
///
/// # Errors
///
/// [`Error::Invalid`] when a page is past the table, and whatever [`page_cells`] refuses.
pub(crate) fn page_runs(
    plan: &crate::memory::CachePlan,
    group: &CacheGroupPlan,
    table: &crate::memory::prefix_cache::block_tables::GroupTable,
    num_units: u32,
    first_page: u32,
    pages: u32,
) -> Result<(u32, Runs)> {
    let mut runs: Vec<(usize, Vec<i64>)> = Vec::new();
    let mut rows = 0u32;
    for page in first_page..first_page.saturating_add(pages) {
        let position = usize::try_from(page.saturating_sub(table.start_page))
            .map_err(|_| Error::Invalid("a state page is past its block table".to_owned()))?;
        if position >= table.units.len() / table.shape.units_per_page as usize {
            return Err(Error::Invalid(format!(
                "a state page of {pages} pages from {first_page} is past its block table"
            )));
        }
        let units = table.row(position);
        for checkpoint in &plan.checkpoints {
            for (field, cells) in page_cells(group, plan.columns, num_units, checkpoint, &units)? {
                rows = rows.checked_add(cells.len() as u32).ok_or_else(|| {
                    Error::Invalid("a state page holds too many blocks".to_owned())
                })?;
                match runs.last_mut() {
                    Some((held, run)) if *held == field => run.extend(cells),
                    _ => runs.push((field, cells)),
                }
            }
        }
    }
    Ok((rows, runs))
}

/// The cells of one storage field, in row order.
pub(crate) type Runs = Vec<(usize, Vec<i64>)>;

/// A page's blocks as a destination writes them: the rows a transfer states and the cells of this
/// rank's own units they land in.
///
/// It is what both directions of an import compare: the transfer states `rows` and `width`, this
/// states the same two figures from the destination's own units of the same pages, and the
/// scatter writes the rows into `runs` once they agree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatePage {
    /// Blocks the page holds.
    pub rows: u32,
    /// Bytes one block holds.
    pub width: u64,
    /// The cells of the destination's units, per storage field, in row order.
    pub runs: Vec<(usize, Vec<i64>)>,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn group(columns: u32, units_per_page: u32, layers: usize) -> CacheGroupPlan {
        CacheGroupPlan {
            window: None,
            state: true,
            draft: false,
            block_size: 64,
            units_per_page,
            layers: (0..layers).map(|layer| format!("layer.{layer}")).collect(),
            layer_ids: (0..layers as u32).collect(),
            axis: crate::memory::prefix_cache::plan::GroupAxis {
                offset: 0,
                total: columns,
            },
            num_kv_heads: 0,
            total_kv_heads: 0,
            kv_head_offset: 0,
            head_dim: 0,
            dtype: crate::memory::allocation::ElementType::U8,
            compute_dtype: crate::memory::allocation::ElementType::U8,
        }
    }

    fn checkpoint(first_plane: u32, planes: u32) -> Checkpoint {
        Checkpoint {
            first_plane,
            planes,
            fields: Vec::new(),
        }
    }

    /// A page's planes map to the pool's cells the way the retained manifest addresses them:
    /// `column * num_units + unit`, one run per storage field, in plane order.
    #[test]
    fn a_pages_planes_map_to_their_cells_in_field_runs() -> Result<()> {
        // Two columns and one unit per page: a page holds two planes, one per column, and the
        // next two planes belong to the second field.
        let group = group(2, 1, 2);
        let cells = page_cells(&group, 2, 8, &checkpoint(0, 4), &[5])?;
        assert_eq!(cells, vec![(0, vec![5, 8 + 5]), (1, vec![5, 8 + 5]),]);
        Ok(())
    }

    /// The fields `page_cells` names are exactly the storages the plan asks for: no state plane
    /// reaches past the last one the pool allocates, which is what `state_fields` exists to say.
    #[test]
    fn the_fields_a_page_names_are_the_storages_the_plan_asks_for() -> Result<()> {
        let state = group(2, 1, 2);
        let plan = crate::memory::prefix_cache::plan::CachePlan {
            columns: 2,
            plane_bytes: 16,
            quantized: false,
            unit_bytes: 64,
            dtype: ipc::PrefixCacheDtype::Float8E4m3Fn,
            groups: vec![state.clone()],
            states: Vec::new(),
            checkpoints: vec![checkpoint(0, 5)],
        };
        // Two columns and one unit per page hold two planes per field, and five planes reach the
        // third field.
        assert_eq!(plan.state_fields(), 3);
        let fields = page_cells(&state, plan.columns, 8, &plan.checkpoints[0], &[5, 6])?;
        let last = fields
            .iter()
            .map(|(field, _)| *field)
            .max()
            .expect("five planes name a field");
        assert!(
            last < plan.state_fields() as usize,
            "a page names field {last} while the pool allocates {}",
            plan.state_fields()
        );
        Ok(())
    }

    /// The row's property on a device: a page's planes gather into the snapshot completely before
    /// any target block is written, so a page whose source units are also its targets comes out a
    /// swap and not two copies of one block.
    ///
    /// This is the reason the row exists. A copy that walked block by block, reading and writing
    /// as it went, would report success on every non-overlapping page and silently lose state on
    /// exactly the pages a prefix cache shares between an old and a new request.
    ///
    /// Run it on a GPU host, alone on the device:
    ///
    /// ```text
    /// cargo test -p executor --lib state::tests -- --ignored --test-threads=1
    /// ```
    ///
    /// It needs `nvcc` on `PATH`; `UNISERVE_PROBE_ARCH` names the architecture (default `sm_100a`).
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_state_page_whose_units_overlap_its_targets_comes_out_correct() {
        use platform::{
            Allocator, Budget, Device, DeviceId, Function, LaunchConfig, Library, PinnedBuffer,
            PoolKind, Stream, memcpy_async,
        };
        use std::path::PathBuf;
        use std::process::Command;

        const DEVICE: DeviceId = DeviceId(0);
        const BLOCKS: usize = 8;
        const WIDTH: usize = 16;

        #[derive(Default)]
        struct Allowance;
        impl Budget for Allowance {
            fn book(
                &self,
                _device: DeviceId,
                _pool: PoolKind,
                _bytes: u64,
            ) -> platform::Result<()> {
                Ok(())
            }
        }

        let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/state_copy.cu");
        let cubin = std::env::temp_dir().join(format!("state-copy-{}.cubin", std::process::id()));
        let arch = std::env::var("UNISERVE_PROBE_ARCH").unwrap_or_else(|_| "sm_100a".to_owned());
        let status = Command::new("nvcc")
            .arg("-cubin")
            .arg(format!("-arch={arch}"))
            .arg("-o")
            .arg(&cubin)
            .arg(&source)
            .status()
            .expect("this probe compiles its kernels with nvcc, which must be on PATH");
        assert!(status.success(), "nvcc failed on {}", source.display());

        let device = Device::new(DEVICE).expect("a CUDA device");
        let mut allocator =
            Allocator::new(&device, std::sync::Arc::new(Allowance)).expect("an allocator");
        allocator
            .reserve(PoolKind::Graph, 1 << 20)
            .expect("reserve the probe's pool");
        let blocks = BLOCKS * WIDTH;
        let field = allocator
            .allocate(PoolKind::Graph, blocks)
            .expect("the field's planes");
        let snapshot = allocator
            .allocate(PoolKind::Graph, blocks)
            .expect("the snapshot's rows");
        // Two index vectors: the page read and the page written, which is the page read reversed.
        let read = allocator
            .allocate(PoolKind::Graph, 2 * 8)
            .expect("the source page's cells");
        let write = allocator
            .allocate(PoolKind::Graph, 2 * 8)
            .expect("the target page's cells");
        let stream = Stream::new(DEVICE).expect("a stream");
        let image = std::fs::read(&cubin).expect("the compiled probe kernels");
        let library = Library::load(&device, &image).expect("load the probe kernels");
        let read_kernel = library.kernel("state_snapshot").expect("state_snapshot");
        let write_kernel = library.kernel("state_scatter").expect("state_scatter");

        // Block `k` holds `k * 10 + j` at byte `j`, so a swap is visible byte by byte.
        let mut fill = PinnedBuffer::new(DEVICE, blocks).expect("staging");
        // SAFETY: the pinned buffer is the only writer of these bytes and holds
        // `blocks` of them, which is the whole field.
        for (index, byte) in unsafe { fill.as_mut_slice() }.iter_mut().enumerate() {
            *byte = u8::try_from(index / WIDTH * 10 + index % WIDTH).expect("a byte");
        }
        // SAFETY: both ranges are the same `blocks` bytes, in buffers of their
        // own, and the copy is ordered on `stream`.
        unsafe { memcpy_async(field.address(), fill.address(), blocks, &stream) }
            .expect("fill the field");
        // Page A reads blocks 3 and 4; page B writes block 4 and then block 3.
        for (buffer, cells) in [(&read, [3i64, 4]), (&write, [4i64, 3])] {
            let mut stage = PinnedBuffer::new(DEVICE, 16).expect("staging");
            // SAFETY: the staging buffer holds 16 bytes and the two cells below
            // fill exactly them.
            let bytes = unsafe { stage.as_mut_slice() };
            for (index, cell) in cells.iter().enumerate() {
                bytes[index * 8..index * 8 + 8].copy_from_slice(&cell.to_ne_bytes());
            }
            // SAFETY: the cell buffer holds these two `i64` indices and the
            // staging buffer holds their 16 bytes; the copy is ordered on
            // `stream`, so the launch below reads them after it.
            unsafe { memcpy_async(buffer.address(), stage.address(), 16, &stream) }
                .expect("write the cells");
        }
        stream.synchronize().expect("order the fills");

        let launch = |kernel: &Function, destination: u64, cells: u64, source: u64| {
            let mut rows = 2i32;
            let mut width = WIDTH as i32;
            let mut arguments: [*mut std::ffi::c_void; 5] = [
                (destination as *mut std::ffi::c_void),
                (cells as *mut std::ffi::c_void),
                (source as *mut std::ffi::c_void),
                (&raw mut rows).cast(),
                (&raw mut width).cast(),
            ];
            kernel
                .launch_ex(
                    &stream,
                    &LaunchConfig::new([2, 1, 1], [WIDTH as u32, 1, 1]),
                    &mut arguments,
                )
                .expect("launch the copy");
        };
        // Gather every source block, then write every target block: the two launches the row
        // guarantees, in this order and no other.
        launch(
            &read_kernel,
            snapshot.address(),
            read.address(),
            field.address(),
        );
        launch(
            &write_kernel,
            field.address(),
            write.address(),
            snapshot.address(),
        );
        stream.synchronize().expect("order the copies");

        let back = PinnedBuffer::new(DEVICE, blocks).expect("staging");
        // SAFETY: the read-back buffer holds `blocks` bytes and the field holds
        // the same, on one device; the copy is ordered on `stream` after both
        // launches.
        unsafe { memcpy_async(back.address(), field.address(), blocks, &stream) }
            .expect("read the field back");
        stream.synchronize().expect("synchronize");
        // SAFETY: the staging copy above has completed (`synchronize` below the
        // launches) and nothing writes these bytes while they are read.
        let bytes = unsafe { back.as_slice() };
        let block = |index: usize| bytes[index * WIDTH..(index + 1) * WIDTH].to_vec();
        let pattern = |index: usize| -> Vec<u8> {
            (0..WIDTH)
                .map(|j| u8::try_from(index * 10 + j).expect("a byte"))
                .collect()
        };
        assert_eq!(block(3), pattern(4), "block 3 did not take block 4's bytes");
        assert_eq!(block(4), pattern(3), "block 4 did not take block 3's bytes");
    }

    /// A page of two units asks the two units it was given, so a checkpoint that needs more
    /// planes than the page's units and fields hold is refused rather than wrapped.
    #[test]
    fn a_page_naming_fewer_units_than_its_planes_need_is_refused() {
        let group = group(2, 2, 4);
        // Four planes span two units of two columns; one unit cannot hold the second half.
        assert!(page_cells(&group, 2, 8, &checkpoint(0, 4), &[3]).is_err());
        // Two units hold the page's first field's four planes.
        assert_eq!(
            page_cells(&group, 2, 8, &checkpoint(0, 4), &[3, 4]).expect("two units"),
            vec![(0, vec![3, 8 + 3, 4, 8 + 4])]
        );
    }
}
