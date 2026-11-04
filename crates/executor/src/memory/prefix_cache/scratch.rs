//! The scratch page set the paired forward writes: 7.3's shadow step must not
//! touch the resident cache, so the copy of a layer's K/V a shadow batch
//! writes lives here.
//!
//! The region is one allocation from the device's `Scratch` pool, sized once
//! from the plan and one batch's largest page set, and reused by every paired
//! forward - it is never exported, imported or parked, which is why it lives
//! inside the device's allowance rather than on the Storage Server's retained
//! storage as the resident planes do.
//!
//! Addressing: row `r` of a batch owns pages `r * pages .. (r + 1) * pages` of
//! the set, and the row's block table is those page indices, so two rows of one
//! batch never write the same page.

use std::sync::Arc;

use platform::{Buffer, DeviceId, PoolKind};

use super::LayerPlanes;
use super::plan::CachePlan;
use super::{PrefixCache, Result};
use crate::arena::Pools;
use crate::memory::allocation::{Backing, ElementType, MemoryKind, StorageView};

/// Device memory one pool owns, viewed as a backing: the block belongs to the
/// device's allocator, so dropping this view releases nothing, and the pool
/// frees the block when the whole region goes.
pub(crate) struct PooledBuffer {
    buffer: Buffer,
}

impl PooledBuffer {
    fn new(buffer: Buffer) -> Arc<Self> {
        Arc::new(Self { buffer })
    }

    fn view(self: &Arc<Self>, dtype: ElementType, shape: &[i64]) -> Result<StorageView> {
        StorageView::contiguous(Arc::clone(self) as Arc<dyn Backing>, 0, dtype, shape)
    }
}

impl Backing for PooledBuffer {
    fn device(&self) -> MemoryKind {
        MemoryKind::Cuda(self.buffer.device())
    }

    fn address(&self) -> u64 {
        self.buffer.address()
    }

    fn bytes(&self) -> usize {
        self.buffer.len()
    }
}

/// One batch's scratch pages, mapped from the device's `Scratch` pool.
pub struct ScratchCache {
    plan: Arc<CachePlan>,
    /// Pages one row of the set owns.
    pages: u32,
    /// Units one page of the widest group occupies.
    units_per_page: u32,
    /// The units the whole region holds, over every column.
    num_units: u32,
    /// One `[rows, pages]` block table per row, as the attention kernel reads
    /// it: page `k` of row `r` is `r * pages + k`.
    tables: Vec<Arc<PooledBuffer>>,
    values: [Arc<PooledBuffer>; 2],
    initialized: [Arc<PooledBuffer>; 2],
    scales: Option<[Arc<PooledBuffer>; 2]>,
}

impl ScratchCache {
    /// Take the region for `rows` rows of `pages` pages from `pools`, sized by
    /// `plan`: both value planes, the flag plane of each field, and the scale
    /// plane of each on a quantized pool.
    ///
    /// `pages` is the largest page set one row of a paired batch can need; a
    /// later request for more is refused by name rather than truncated.
    pub fn allocate(
        plan: &Arc<CachePlan>,
        rows: u32,
        pages: u32,
        device: DeviceId,
        pools: &mut Pools,
    ) -> Result<Self> {
        if rows == 0 || pages == 0 || plan.columns == 0 {
            return Err(crate::Error::Invalid(
                "a scratch page set needs rows, pages and columns".into(),
            ));
        }
        let units_per_page = plan
            .groups
            .iter()
            .map(|group| group.units_per_page)
            .max()
            .unwrap_or(1)
            .max(1);
        let num_units = pages
            .checked_mul(units_per_page)
            .ok_or_else(|| crate::Error::Invalid("a scratch page set overflows units".into()))?;
        let cells = u64::from(num_units).max(1);

        let mut take = |dtype: ElementType, elements: u64| -> Result<Arc<PooledBuffer>> {
            let bytes = elements * dtype.bytes() as u64;
            let buffer = pools.allocate(PoolKind::Scratch, bytes)?;
            Ok(PooledBuffer::new(buffer))
        };

        let values = [
            take(ElementType::U8, cells * plan.plane_bytes)?,
            take(ElementType::U8, cells * plan.plane_bytes)?,
        ];
        let initialized = [take(ElementType::U8, cells)?, take(ElementType::U8, cells)?];
        let scales = plan
            .quantized
            .then(|| -> Result<[Arc<PooledBuffer>; 2]> {
                Ok([
                    take(ElementType::F32, cells)?,
                    take(ElementType::F32, cells)?,
                ])
            })
            .transpose()?;

        let per_table = u64::from(pages) * 4;
        let mut tables = Vec::with_capacity(rows as usize);
        for _ in 0..rows {
            tables.push(PooledBuffer::new(
                pools.allocate(PoolKind::Scratch, per_table)?,
            ));
        }

        let _ = device;
        Ok(Self {
            plan: Arc::clone(plan),
            pages,
            units_per_page,
            num_units,
            tables,
            values,
            initialized,
            scales,
        })
    }

    /// The pages a batch of `rows` rows may use.
    pub fn pages(&self) -> u32 {
        self.pages
    }

    /// Row `row`'s block table over the set: page `k` of the row is
    /// `row * pages + k`, and a row with no scratch (a padded row) is refused.
    pub fn table(&self, row: u32) -> Result<StorageView> {
        let buffer = self
            .tables
            .get(row as usize)
            .ok_or_else(|| crate::Error::Invalid(format!("the scratch set holds no row {row}")))?;
        buffer.view(ElementType::I32, &[i64::from(self.pages)])
    }

    /// One layer's planes of `group`, over the scratch pages: the same
    /// geometry the resident pool states, with `pages` pages of this set.
    pub fn layer_planes(&self, group: usize, layer: usize) -> Result<LayerPlanes> {
        let plan = self.plan.groups.get(group).ok_or_else(|| {
            crate::Error::Invalid(format!("cache group {group} is not in the plan"))
        })?;
        if layer >= plan.layers.len() {
            return Err(crate::Error::Invalid(format!(
                "cache group {group} holds {} layers, not {layer}",
                plan.layers.len()
            )));
        }
        let columns = self.plan.columns.max(1);
        let column = (layer as u32) % columns;
        let unit = (layer as u32) / columns;
        if unit >= self.units_per_page {
            return Err(crate::Error::Invariant(format!(
                "layer {layer} of cache group {group} lies outside a scratch page"
            )));
        }
        let dtype = plan.dtype;
        let elements_per_unit = i64::try_from(self.plan.plane_bytes / dtype.bytes() as u64)
            .map_err(|_| crate::Error::Invariant("a unit of a plane overflows".into()))?;
        let heads = i64::from(plan.num_kv_heads);
        let dim = i64::from(plan.head_dim);
        let offset = (i64::from(column) * i64::from(self.num_units) + i64::from(unit))
            * elements_per_unit
            * dtype.bytes() as i64;
        let values = |field: usize| -> Result<StorageView> {
            StorageView::strided(
                Arc::clone(&self.values[field]) as Arc<dyn Backing>,
                usize::try_from(offset).map_err(|_| {
                    crate::Error::Invariant("a scratch view lies outside its storage".into())
                })?,
                dtype,
                &[
                    i64::from(self.pages),
                    i64::from(plan.block_size),
                    heads,
                    dim,
                ],
                &[
                    i64::from(self.units_per_page) * elements_per_unit,
                    heads * dim,
                    dim,
                    1,
                ],
            )
        };
        let plane = |buffers: &[Arc<PooledBuffer>; 2],
                     field: usize,
                     dtype: ElementType|
         -> Result<StorageView> {
            let cells = i64::from(column) * i64::from(self.num_units) + i64::from(unit);
            let offset = usize::try_from(cells * dtype.bytes() as i64).map_err(|_| {
                crate::Error::Invariant("a scratch view lies outside its storage".into())
            })?;
            StorageView::strided(
                Arc::clone(&buffers[field]) as Arc<dyn Backing>,
                offset,
                dtype,
                &[i64::from(self.pages), 1, 1, 1],
                &[i64::from(self.units_per_page), 1, 1, 1],
            )
        };
        Ok(LayerPlanes {
            block_size: plan.block_size,
            pages: self.pages,
            dtype: dtype.to_ops().ok_or_else(|| {
                crate::Error::Invalid(format!(
                    "cache group {group} holds storage element {dtype:?}, which no kernel operand type describes"
                ))
            })?,
            key: values(0)?,
            value: values(1)?,
            key_scale: self
                .scales
                .as_ref()
                .map(|scales| plane(scales, 0, ElementType::F32))
                .transpose()?,
            value_scale: self
                .scales
                .as_ref()
                .map(|scales| plane(scales, 1, ElementType::F32))
                .transpose()?,
            key_initialized: plane(&self.initialized, 0, ElementType::U8)?,
            value_initialized: plane(&self.initialized, 1, ElementType::U8)?,
        })
    }

    /// The set's key and value scales of `group`, as the paired forward reads
    /// them; `None` on a dense pool.
    pub fn scales(&self, _group: usize) -> Option<(StorageView, StorageView)> {
        let scales = self.scales.as_ref()?;
        let key = scales[0]
            .view(ElementType::F32, &[i64::from(self.num_units)])
            .ok()?;
        let value = scales[1]
            .view(ElementType::F32, &[i64::from(self.num_units)])
            .ok()?;
        Some((key, value))
    }
}

impl PrefixCache {
    /// The scratch page set a paired forward writes: `rows` rows of `pages`
    /// pages each, taken from the device's `Scratch` pool.
    pub fn scratch(
        &self,
        rows: u32,
        pages: u32,
        device: DeviceId,
        pools: &mut Pools,
    ) -> Result<ScratchCache> {
        ScratchCache::allocate(self.plan(), rows, pages, device, pools)
    }
}
