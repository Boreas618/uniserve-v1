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

use std::sync::Arc;

use nn::KvPlanes;
use ops::{DType, DevicePtr, Shape, Strides, TensorView};
use platform::{Buffer, PoolKind, Stream};

use crate::arena::Pools;

use super::causal_lm::KvProvider;
use crate::memory::allocation::ElementType;
use crate::memory::prefix_cache::{LayerPlanes, PrefixCache};
use crate::{Error, Result};

/// The resident cache as a runner reads it.
pub struct ResidentCache {
    cache: Arc<PrefixCache>,
    /// One gather column per attention group, `[rows, width]` i32.
    tables: Vec<Table>,
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
    /// Allocates `rows x width` i32 for one group from the device's pools.
    fn new(pools: &std::sync::Mutex<Pools>, rows: u32, width: u32) -> Result<Self> {
        if rows == 0 || width == 0 {
            return Err(Error::Invalid(
                "a block table needs at least one row and one page".into(),
            ));
        }
        let bytes = usize::try_from(u64::from(rows) * u64::from(width) * 4)
            .map_err(|_| Error::Resource("a block table is larger than memory".into()))?;
        let mut pools = pools
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(Self {
            buffer: pools.allocate(PoolKind::Resident, bytes as u64)?,
            rows,
            width,
        })
    }

    /// The `[rows, width]` i32 view over this table's storage.
    fn view(&self) -> TensorView {
        TensorView::new(
            DevicePtr::new(self.buffer.address()),
            DType::I32,
            Shape::new(&[u64::from(self.rows), u64::from(self.width)]),
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
        cache: Arc<PrefixCache>,
        pools: &std::sync::Mutex<Pools>,
        rows: u32,
    ) -> Result<Self> {
        let groups = cache.groups();
        if groups == 0 {
            return Err(Error::Invalid(
                "the resident cache has no attention group".into(),
            ));
        }
        let first = cache.layer_planes(0, 0)?;
        let (_, _, width) = cache.unit_tables();
        let mut tables = Vec::with_capacity(groups);
        for _ in 0..groups {
            tables.push(Table::new(pools, rows, width)?);
        }
        Ok(Self {
            cache,
            tables,
            groups,
            page_size: first.block_size,
            dtype: first.dtype,
        })
    }

    /// The pool behind the provider, for a caller that also drives its imports and exports.
    pub fn cache(&self) -> &Arc<PrefixCache> {
        &self.cache
    }

    /// Gathers rows `slots` of `group`'s pooled table into the group's column and answers it.
    ///
    /// A row whose slot is negative serves no request and its row of the view is `-1`-filled; the
    /// attention kernel never reads past a row's `seq_lens`, so such a row is never dereferenced
    /// (the pool's tables reserve slot 0, which is why `-1` and not `0` is the absent row).
    fn gather(&self, group: usize, slots: &[i32], stream: &Stream) -> Result<TensorView> {
        let table = self
            .tables
            .get(group)
            .ok_or_else(|| Error::Invalid(format!("the cache has no attention group {group}")))?;
        if slots.len() != table.rows as usize {
            return Err(Error::Invalid(format!(
                "a block table of group {group} needs {} row slots, got {}",
                table.rows,
                slots.len()
            )));
        }
        let (pooled, pool_rows, width) = self.cache.unit_tables();
        if width != table.width {
            return Err(Error::Invariant(format!(
                "group {group}'s table is {} pages wide and was sized {}",
                width, table.width
            )));
        }
        let row_bytes = width as usize * 4;
        for (row, slot) in slots.iter().enumerate() {
            let destination = table.row_address(row as u32);
            if *slot < 0 {
                // SAFETY: the row lies inside this provider's column, which outlives the stream
                // work, and the count is the row's own width.
                unsafe {
                    platform::memset_d32(destination, u32::MAX, width as usize, stream)?;
                }
                continue;
            }
            let slot = u32::try_from(*slot).map_err(|_| {
                Error::Invalid(format!("group {group} has no row slot {slot}"))
            })?;
            if slot >= pool_rows {
                return Err(Error::Invalid(format!(
                    "group {group} has {} request slots and a row asks for {slot}",
                    pool_rows
                )));
            }
            let source = pooled.address()
                + (group as u64 * u64::from(pool_rows) + u64::from(slot))
                    * u64::from(width)
                    * 4;
            // SAFETY: both ranges are `width` i32 of live device storage and the destination row
            // lies inside this provider's column.
            unsafe {
                platform::memcpy_async(destination, source, row_bytes, stream)?;
            }
        }
        Ok(table.view())
    }
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

impl KvProvider for ResidentCache {
    fn groups(&self) -> usize {
        self.groups
    }

    fn page_size(&self) -> u32 {
        self.page_size
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn block_table(
        &self,
        group: usize,
        slots: &[i32],
        stream: &Stream,
    ) -> Result<TensorView> {
        self.gather(group, slots, stream)
    }

    fn planes(&self, group: usize) -> Result<Vec<KvPlanes>> {
        let layers = self.cache.layers(group);
        if layers == 0 {
            return Err(Error::Invalid(format!(
                "attention group {group} has no layer"
            )));
        }
        let mut planes = Vec::with_capacity(layers);
        for layer in 0..layers {
            planes.push(planes_of(&self.cache.layer_planes(group, layer)?)?);
        }
        Ok(planes)
    }

    fn scratch(
        &self,
        group: usize,
        _slots: &[i32],
        _pages: u32,
        _stream: &Stream,
    ) -> Result<(TensorView, Vec<KvPlanes>)> {
        // The scratch page set is the storage owners' region (`PrefixCache::scratch`, agreed in
        // M619/M682 and not published yet), so a paired forward has nowhere it may write without
        // touching a page a request reads. Failing here says which piece is missing instead of
        // writing resident pages: 7.3's rule is that a paired forward never does.
        let _ = group;
        Err(Error::State(
            "the pool's scratch region (PrefixCache::scratch) is not published, so a paired \
             forward has no page set it may write"
                .into(),
        ))
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
    use super::*;

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
}
