//! `KvProvider` over the resident prefix cache (4.5, section 8).
//!
//! A runner reads its KV cache through [`KvProvider`](super::causal_lm::KvProvider) and never
//! touches the pool itself, so the pool's layout stays the storage owners' business. This is the
//! provider a serving rank hands its runners: it answers the resident cache's pages, tables and
//! scales, and a scratch page set for the paired forward (7.3).
//!
//! The block table is the one thing the provider builds rather than borrows. A pooled block table
//! holds every request slot's pages (`[table = group, slot, width]`, the storage owner's layout);
//! a batch reads the rows of the slots it serves, so the provider gathers those rows into a
//! per-group column of its own, on the runner's stream, and answers a `[rows, width]` view the
//! captured graph can address. The column is allocated when the provider is built, for the largest
//! bucket, so a step allocates nothing (4.5).

use std::sync::{Arc, Mutex};

use nn::KvPlanes;
use ops::{DType, DevicePtr, Shape, Strides, TensorView};
use platform::{Buffer, PoolKind, Stream};

use crate::arena::Pools;
use crate::memory::prefix_cache::scratch::ScratchCache;

use super::causal_lm::KvProvider;
use crate::memory::prefix_cache::{LayerPlanes, PrefixCache};
use crate::{Error, Result};

/// The resident cache as a runner reads it.
pub struct ResidentCache {
    /// The pool. A provider only reads it, but the storage owners mutate it (installs, released
    /// slots, applied transfers) while a batch may be gathering its table, so the share is behind
    /// the owners' one lock rather than an unsynchronized `Arc`.
    cache: Arc<Mutex<PrefixCache>>,
    /// The device the cache and the scratch set live on.
    device: platform::DeviceId,
    /// The device's pools, attached by the executor before the first gather: the scratch page set
    /// is mapped through them, so the mapping is inside the device's one book (4.1).
    pools: Mutex<Option<Arc<Mutex<Pools>>>>,
    /// The paired forward's page set, mapped on the first shadow walk. One set serves every group:
    /// its rows are the batch's rows and its pages are the largest page set a row needs.
    scratch: Mutex<Option<Scratch>>,
    /// One gather column per attention group, `[rows, width]` i32, carved when the device's pools
    /// are attached (the executor creates them, so they cannot be had at construction).
    tables: Vec<Table>,
    /// The rows the columns are sized for: the largest bucket's.
    rows: u32,
    /// The pages a row of the pooled table holds (`unit_tables`' width).
    width: u32,
    groups: usize,
    page_size: u32,
    dtype: DType,
}

/// One group's gathered block table: the device storage and the shape of the view over it.
///
/// The storage is carved from the device's one pool set (`PoolKind::Resident`), never mapped
/// directly: a mapping outside that allocator would not be booked against the device's budget
/// (4.1).
struct Table {
    buffer: Buffer,
    rows: u32,
    width: u32,
}

impl Table {
    /// Writes `indices` (row-major, `rows * width` values) into the table once, from the host.
    ///
    /// A block table states fixed page ids for the runner's lifetime, so this runs once per table
    /// and never on a step: a paired forward reads the ids, it does not build them.
    fn fill(&self, indices: &[i32], stream: &Stream) -> Result<()> {
        let bytes = std::mem::size_of_val(indices);
        if bytes != self.buffer.len() {
            return Err(Error::Invariant(format!(
                "a block table of {} bytes cannot hold {bytes} of indices",
                self.buffer.len()
            )));
        }
        let mut host = platform::PinnedBuffer::new(self.buffer.device(), bytes)?;
        // SAFETY: the source is `indices`, `bytes` long, and the destination is this table's own
        // host buffer, allocated for exactly that length and not read concurrently.
        unsafe {
            host.as_mut_slice()[..bytes]
                .copy_from_slice(std::slice::from_raw_parts(indices.as_ptr().cast::<u8>(), bytes));
            // SAFETY: both addresses are this table's device buffer and that host buffer, of the
            // same length; the copy is stream-ordered before the walk that reads the table.
            platform::memcpy_async(self.buffer.address(), host.address(), bytes, stream)?;
        }
        Ok(())
    }

    /// Allocates `rows x width` i32 for one group from the device's pools.
    fn new(pools: &mut Pools, rows: u32, width: u32) -> Result<Self> {
        if rows == 0 || width == 0 {
            return Err(Error::Invalid(
                "a block table needs at least one row and one page".into(),
            ));
        }
        let bytes = usize::try_from(u64::from(rows) * u64::from(width) * 4)
            .map_err(|_| Error::Resource("a block table is larger than memory".into()))?;
        Ok(Self {
            buffer: pools.allocate(PoolKind::Resident, bytes as u64)?,
            rows,
            width,
        })
    }

    /// The `[rows, width]` i32 view over this table's storage.
    fn view(&self) -> TensorView {
        self.view_rows(self.rows)
    }

    /// The `[rows, width]` view over the table's first `rows` rows.
    ///
    /// A bucket narrower than the table addresses its own prefix of it: inside a captured graph
    /// every per-sequence operand has the bucket's row count, and the page table is one of them -
    /// so the table's storage is the generation's widest bucket while the view is the bucket'a own.
    fn view_rows(&self, rows: u32) -> TensorView {
        TensorView::new(
            DevicePtr::new(self.buffer.address()),
            DType::I32,
            Shape::new(&[u64::from(rows), u64::from(self.width)]),
        )
    }

    /// Where row `row` starts.
    fn row_address(&self, row: u32) -> u64 {
        self.buffer.address() + u64::from(row) * u64::from(self.width) * 4
    }
}

impl ResidentCache {
    /// A provider over `cache` for a runner whose largest bucket has `rows` rows.
    ///
    /// `rows` is the bucket's row count, padding included: every captured graph addresses a table
    /// of exactly that height, so the gather column is allocated once, here, and never grows.
    pub fn new(
        cache: Arc<Mutex<PrefixCache>>,
        device: platform::DeviceId,
        rows: u32,
    ) -> Result<Self> {
        let groups = {
            let cache = cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            cache.groups()
        };
        if groups == 0 {
            return Err(Error::Invalid(
                "the resident cache has no attention group".into(),
            ));
        }
        let (first, width) = {
            let cache = cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (cache.layer_planes(0, 0)?, cache.unit_tables().2)
        };
        Ok(Self {
            cache,
            device,
            pools: Mutex::new(None),
            scratch: Mutex::new(None),
            tables: Vec::new(),
            rows,
            width,
            groups,
            page_size: first.block_size,
            dtype: first.dtype,
        })
    }

    /// The pool behind the provider, for a caller that also drives its imports and exports.
    pub fn cache(&self) -> &Arc<Mutex<PrefixCache>> {
        &self.cache
    }

    /// The paired forward's view: its group's table and the layer planes over the scratch pages.
    fn scratch_view(
        &self,
        group: usize,
        slots: &[i32],
        scratch: &Scratch,
    ) -> (TensorView, Vec<KvPlanes>) {
        let layers = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .layers(group);
        let mut planes = Vec::with_capacity(layers);
        for layer in 0..layers {
            // A layer's planes are borrowed views over the region; a failure here is a region that
            // does not hold the layer, which the provider reports by leaving the plane out.
            match scratch
                .cache
                .layer_planes(group, layer)
                .and_then(|planes| planes_of(&planes))
            {
                Ok(plane) => planes.push(plane),
                Err(error) => {
                    tracing::warn!(%error, group, layer, "a scratch plane does not resolve");
                }
            }
        }
        let rows = slots.len().max(1) as u32;
        match scratch.tables.get(group) {
            Some(table) => (
                TensorView::new(
                    DevicePtr::new(table.buffer.address()),
                    DType::I32,
                    Shape::new(&[u64::from(rows), u64::from(table.width)]),
                ),
                planes,
            ),
            None => (
                TensorView::new(DevicePtr::new(0), DType::I32, Shape::new(&[0, 0])),
                planes,
            ),
        }
    }

    /// Gathers rows `slots` of `group`'s pooled table into the group's column and answers it.
    ///
    /// A row whose slot is negative serves no request and its row of the view is `-1`-filled; the
    /// attention kernel never reads past a row's `seq_lens`, so such a row is never dereferenced
    /// (the pool's tables reserve slot 0, which is why `-1` and not `0` is the absent row).
    fn gather(&self, group: usize, slots: &[i32], stream: &Stream) -> Result<TensorView> {
        if self.tables.is_empty() {
            return Err(Error::State(
                "the cache provider has no gather columns: the executor attaches the device's \
                 pools before the first capture"
                    .into(),
            ));
        }
        let table = self
            .tables
            .get(group)
            .ok_or_else(|| Error::Invalid(format!("the cache has no attention group {group}")))?;
        // A batch narrower than the table's rows is the normal case: the table's storage is sized
        // for the widest batch the generation may see, and a bucket gathers its own prefix of it.
        let rows = row_count(group, table.rows, slots)?;
        let (pooled, pool_rows, width) = {
            let cache = self
                .cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            cache.unit_tables()
        };
        if width != table.width {
            return Err(Error::Invariant(format!(
                "group {group}'s table is {} pages wide and was sized {}",
                width, table.width
            )));
        }
        let row_bytes = width as usize * 4;
        for (row, slot) in slots.iter().enumerate() {
            let slot = *slot;
            let destination = table.row_address(row as u32);
            if slot < 0 {
                // An absent row names **page zero** for every entry, not a sentinel: a vendor
                // kernel forms a page's address before it checks the row's length, and `-1` would
                // put that address a whole page-stride before the pool - an illegal address on a
                // row that is not the batch's last live one. Page zero is the pool's reserved
                // padding unit, so the address is in range and a zero-length row attends to
                // nothing. The row's `seq_lens` entry is the batch's own and states the same zero.
                //
                // SAFETY: the row lies inside this provider's column, which outlives the stream
                // work, and the count is the row's own width.
                unsafe {
                    platform::memset_d32(destination, 0, width as usize, stream)?;
                }
                continue;
            }
            let slot = u32::try_from(slot)
                .map_err(|_| Error::Invalid(format!("group {group} has no row slot {slot}")))?;
            if slot >= pool_rows {
                return Err(Error::Invalid(format!(
                    "group {group} has {} request slots and a row asks for {slot}",
                    pool_rows
                )));
            }
            let source = pooled.address()
                + (group as u64 * u64::from(pool_rows) + u64::from(slot)) * u64::from(width) * 4;
            // SAFETY: both ranges are `width` i32 of live device storage and the destination row
            // lies inside this provider's column.
            unsafe {
                platform::memcpy_async(destination, source, row_bytes, stream)?;
            }
        }
        // The view is the bucket's, not the table's: attention compares it with the batch's own
        // sequence count, and inside a captured graph both are the bucket's rows.
        Ok(table.view_rows(rows))
    }
}


/// The slot each row of a group's table gathers from.
///
/// A row the batch does not fill is `-1` **here**, and its gathered row names page zero: the slot
/// is the cache's own numbering, where a negative slot states that no request owns the row, and the
/// row is then filled with the padding page rather than that slot's pool row. The two are different
/// things - who owns the row, and what the kernel may read - and only the first can be a sentinel.
/// A batch wider than the table is refused, because a table one row short would silently drop a
/// request's pages.
fn row_count(group: usize, table_rows: u32, slots: &[i32]) -> Result<u32> {
    if slots.is_empty() {
        return Err(Error::Invalid(format!(
            "group {group}'s block table was handed no row: a batch has at least one sequence"
        )));
    }
    if slots.len() > table_rows as usize {
        return Err(Error::Invalid(format!(
            "a block table of group {group} has {table_rows} row slots and was handed {}",
            slots.len()
        )));
    }
    u32::try_from(slots.len()).map_err(|_| Error::Invalid("a batch's row count overflows".into()))
}

/// One layer's borrowed planes as the model's cache reads them.
fn planes_of(layer: &LayerPlanes) -> Result<KvPlanes> {
    Ok(KvPlanes {
        key: view_of(&layer.key)?,
        value: view_of(&layer.value)?,
        key_initialized: view_of(&layer.key_initialized)?,
        value_initialized: view_of(&layer.value_initialized)?,
        key_scale: layer.key_scale.as_ref().map(view_of).transpose()?,
        value_scale: layer.value_scale.as_ref().map(view_of).transpose()?,
    })
}

/// A borrowed storage view as a kernel operand view, strides included.
///
/// A plane is strided on purpose (a page's layers are columns of one storage), so the strides are
/// carried over rather than recomputed row-major.
fn view_of(view: &crate::memory::allocation::StorageView) -> Result<TensorView> {
    let dtype = view.dtype().to_ops().ok_or_else(|| {
        Error::Invalid(format!(
            "a cache plane of {:?} has no kernel operand type",
            view.dtype()
        ))
    })?;
    let shape: Vec<u64> = view
        .shape()
        .iter()
        .map(|axis| u64::try_from(*axis).unwrap_or(0))
        .collect();
    let strides: Vec<u64> = view
        .strides()
        .iter()
        .map(|stride| u64::try_from(*stride).unwrap_or(0))
        .collect();
    Ok(TensorView::with_strides(
        DevicePtr::new(view.address()),
        dtype,
        Shape::new(&shape),
        Strides::new(&strides),
    ))
}

/// The paired forward's scratch page set: the region and one `[rows, pages]` i32 block table per
/// group over it.
///
/// The set is the shadow's own - its pages are never a serving request's (7.3) - so the table's
/// page ids are this set's own numbering: page `k` of row `r` is `r * pages + k`, which is what the
/// pool's per-row tables say and what the attention kernel reads.
struct Scratch {
    cache: ScratchCache,
    tables: Vec<Table>,
}

impl KvProvider for ResidentCache {
    fn cache(&self) -> Option<&Arc<Mutex<PrefixCache>>> {
        Some(&self.cache)
    }

    /// Writes each slot row into the cache's pooled table, on the batch's stream.
    ///
    /// `PrefixCache::install` validates the tables (the slot inside the request pool and not the
    /// reserved row 0, a known group, units that fit the row's width and the group's page geometry)
    /// and writes every changed row at full width - which also clears the tail of a longer previous
    /// row - with the row's start page and token extent. A table that does not validate is refused
    /// there; nothing here second-guesses it.
    ///
    /// This is the link whose absence made a first batch silently wrong: the gather reads the
    /// pooled table's row for a slot, and no serving path wrote one before this call.
    fn install_block_tables(
        &mut self,
        tables: &[ipc::BlockTable],
        stream: &Stream,
    ) -> Result<()> {
        if tables.is_empty() {
            return Ok(());
        }
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.install(tables, stream)
    }

    /// Applies what the mechanism allocated for a batch's pages: the units become this cache's, then
    /// the leased prefix's copies are issued on the batch's stream.
    ///
    /// The order is the only order that is correct. A destination is a unit of *this* cache, so it
    /// is taken first - `reset_units` clears the previous owner's values and flags, which is what
    /// makes the page new - and the copies come second. They read the sources the mechanism still
    /// leases, and they are issued on the batch's stream, so the forward that reads the destinations
    /// runs after them by stream order and takes no barrier.
    ///
    /// Three things are refused **by name**, because a reader of the deployment's log needs the unit
    /// and the reason rather than the fact that a seam existed: a unit outside the pool's range, a
    /// group this pool does not hold, and a copy whose destination this allocation did not name
    /// (a destination the cache never took is not a destination).
    fn apply_cache_units(
        &mut self,
        units: &[ipc::CacheUnitAllocation],
        stream: &Stream,
    ) -> Result<()> {
        if units.is_empty() {
            return Ok(());
        }
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let groups = cache.groups();
        for allocation in units {
            let group = allocation.group_id as usize;
            if group >= groups {
                return Err(Error::Invalid(format!(
                    "the mechanism allocated units of cache group {group}, but this pool has \
                     {groups} group(s)"
                )));
            }
            let pages = cache.pages(group).map_err(|error| {
                Error::Invalid(format!(
                    "the mechanism allocated units of cache group {group}, which this pool \
                     cannot describe: {error}"
                ))
            })?;
            let held = |unit: ipc::UnitId, what: &str| -> Result<()> {
                // Unit zero is the block tables' sentinel, so a page is one of `1..pages`.
                if unit.0 == 0 || unit.0 >= pages {
                    return Err(Error::Invalid(format!(
                        "the mechanism named cache unit {} of group {group} as {what}, outside \
                         the pool's {} units (unit 0 is the block tables' sentinel)",
                        unit.0, pages
                    )));
                }
                Ok(())
            };
            for &unit in &allocation.unit_ids {
                held(unit, "a new unit")?;
            }
            for &(source, destination) in &allocation.copies {
                held(source, "a copy's source")?;
                held(destination, "a copy's destination")?;
                if !allocation.unit_ids.contains(&destination) {
                    return Err(Error::Invalid(format!(
                        "the mechanism copies into cache unit {} of group {group}, which this \
                         allocation did not name among the units it acquired",
                        destination.0
                    )));
                }
            }
            // The units are this cache's before anything is written into them.
            let acquired: Vec<u32> = allocation.unit_ids.iter().map(|unit| unit.0).collect();
            cache.reset_units(&acquired, stream)?;
            cache.copy_units(&allocation.copies, stream)?;
        }
        Ok(())
    }

    fn attach_pools(&mut self, pools: Arc<std::sync::Mutex<Pools>>) -> Result<()> {
        if !self.tables.is_empty() {
            return Ok(());
        }
        *self
            .pools
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Arc::clone(&pools));
        let mut tables = Vec::with_capacity(self.groups);
        {
            let mut locked = pools
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for _ in 0..self.groups {
                tables.push(Table::new(&mut locked, self.rows, self.width)?);
            }
        }
        self.tables = tables;
        Ok(())
    }

    fn groups(&self) -> usize {
        self.groups
    }

    fn page_size(&self) -> u32 {
        self.page_size
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn block_table(&self, group: usize, slots: &[i32], stream: &Stream) -> Result<TensorView> {
        self.gather(group, slots, stream)
    }

    fn planes(&self, group: usize) -> Result<Vec<KvPlanes>> {
        let cache = self
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let layers = cache.layers(group);
        if layers == 0 {
            return Err(Error::Invalid(format!(
                "attention group {group} has no layer"
            )));
        }
        let mut planes = Vec::with_capacity(layers);
        for layer in 0..layers {
            planes.push(planes_of(&cache.layer_planes(group, layer)?)?);
        }
        Ok(planes)
    }

    fn scratch(
        &self,
        group: usize,
        slots: &[i32],
        pages: u32,
        stream: &Stream,
    ) -> Result<(TensorView, Vec<KvPlanes>)> {
        let rows = slots.len().max(1) as u32;
        let mut guard = self
            .scratch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.is_none() {
            // One set serves every paired forward of this runner: its rows are the largest
            // bucket's and its pages are the walk's, so no shadow step of a later bucket allocates.
            let rows = self.rows.max(rows);
            let pools = self
                .pools
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
                .ok_or_else(|| {
                    Error::State(
                        "the cache provider has no pools: the executor attaches them before the \
                         first capture"
                            .into(),
                    )
                })?;
            let cache = self
                .cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut pools = pools
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // `ScratchCache` states the set's total page count: every row owns `pages` of them,
            // so the set is `rows * pages` pages wide and page `k` of row `r` is `r * pages + k`.
            let total = rows
                .checked_mul(pages)
                .ok_or_else(|| Error::Invalid("a scratch page set overflows pages".into()))?;
            let set = cache.scratch(rows, total, self.device, &mut pools)?;
            // One `[rows, pages]` block table per group, stating the set's own page ids, so two
            // rows of one walk never share a page.
            let mut tables = Vec::with_capacity(self.groups);
            let indices: Vec<i32> = (0..total).map(|page| page as i32).collect();
            for _ in 0..self.groups {
                let table = Table::new(&mut pools, rows, pages)?;
                table.fill(&indices, stream)?;
                tables.push(table);
            }
            *guard = Some(Scratch {
                cache: set,
                tables,
            });
        }
        let scratch = guard.as_ref().ok_or_else(|| {
            Error::Invariant("the scratch page set was allocated and is not held".into())
        })?;
        let asked = rows.saturating_mul(pages);
        if rows > self.rows || asked > scratch.cache.pages() {
            return Err(Error::Invalid(format!(
                "a paired forward asks for {rows} rows of {pages} pages; the scratch set holds {} \
                 rows of {} pages, so the walk would write outside its own pages",
                self.rows,
                scratch.cache.pages() / pages.max(1)
            )));
        }
        Ok(self.scratch_view(group, slots, scratch))
    }
}

impl std::fmt::Debug for ResidentCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResidentCache")
            .field("groups", &self.groups)
            .field("rows", &self.tables.first().map(|table| table.rows))
            .field("page_size", &self.page_size)
            .field("dtype", &self.dtype)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::allocation::ElementType;
    use crate::runners::resident::row_count;

    /// The storage element types a plane can hold and whether they are kernel operand types.
    #[test]
    fn a_plane_of_a_storage_only_type_names_itself() {
        // `ElementType::to_ops` is the one bridge between storage and the kernel ABI; this asserts
        // the provider's error path names the type rather than guessing a mapping.
        for element in [ElementType::Bool, ElementType::F64] {
            assert!(
                element.to_ops().is_none(),
                "{element:?} has no kernel operand type"
            );
            assert!(element.to_ops().is_none());
        }
        for element in [ElementType::BF16, ElementType::F8E4M3, ElementType::I32] {
            assert!(
                element.to_ops().is_some(),
                "{element:?} is a kernel operand type"
            );
        }
    }

    /// A bucket narrower than its group's table is the normal case of a warmup: the table's storage
    /// is the generation's widest batch, and a batch gathers its own prefix. Attention compares the
    /// page table's rows with the batch's own sequence count, so the answer is the batch's count and
    /// not the table's - a table padded to its storage height states sequences the batch has not.
    #[test]
    fn a_batch_narrower_than_its_table_answers_its_own_row_count() {
        assert_eq!(
            row_count(0, 8, &[3, 4, 5, 6]).expect("four rows of an eight-row table"),
            4,
            "a narrower batch answered the table's height rather than its own row count"
        );
        assert_eq!(row_count(0, 2, &[7, 8]).expect("the table's height"), 2);
        assert!(
            row_count(0, 1, &[]).is_err(),
            "a batch of no rows was answered as a table row"
        );
        let error = row_count(0, 4, &[1, 2, 3, 4, 5]).expect_err("five rows of a four-row table");
        assert!(
            error.to_string().contains("has 4 row slots and was handed 5"),
            "the refusal does not name both figures: {error}"
        );
    }

}
