//! The scratch page set the paired forward writes: 7.3's shadow step must not
//! touch the resident cache, so the copy of a layer's K/V a shadow batch
//! writes lives here.
//!
//! The region is one allocation from the device's `Scratch` pool, sized once
//! from the plan and the bytes the device gives the set, and reused by every
//! paired forward - it is never exported, imported or parked, which is why it
//! lives inside the device's allowance rather than on the Storage Server's
//! retained storage as the resident planes do.
//!
//! Addressing: the set's pages are one numbering. A batch whose rows need
//! `pages` pages each gives row `r` the pages `r * pages .. (r + 1) * pages`, so
//! two rows of one batch never write the same page, and a batch of another shape
//! reads the same numbering at another width.

use std::sync::Arc;

use platform::{Buffer, DeviceId, PoolKind};

use super::LayerPlanes;
use super::plan::CachePlan;
use super::{PrefixCache, Result};
use crate::arena::Pools;
use crate::memory::allocation::{Backing, ElementType, MemoryKind, StorageView};

/// The smallest extent the device's pools map, which every allocation of a set
/// rounds up to.
const SCRATCH_GRANULARITY: u64 = 2 << 20;

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
    /// Pages the set holds.
    pages: u32,
    /// Units one page of the widest group occupies.
    units_per_page: u32,
    /// The units the whole region holds, over every column.
    num_units: u32,
    values: [Arc<PooledBuffer>; 2],
    initialized: [Arc<PooledBuffer>; 2],
    scales: Option<[Arc<PooledBuffer>; 2]>,
}

impl ScratchCache {
    /// The pages `bytes` of the `Scratch` pool hold under `plan`: both value
    /// planes, the flag plane of each field, and the scale plane of each on a
    /// quantized pool, over every column, after the granularity each of those
    /// allocations and the set's block tables round up to.
    pub fn pages_within(plan: &CachePlan, bytes: u64, tables: u64) -> u64 {
        let cells = u64::from(Self::units_per_page(plan)) * u64::from(plan.columns.max(1));
        let scales = if plan.quantized { 2 * 4 } else { 0 };
        let page = cells * (2 * plan.plane_bytes + 2 + scales);
        let allocations = 6 + tables;
        bytes
            .saturating_sub(allocations * SCRATCH_GRANULARITY)
            .checked_div(page + 4 * tables)
            .unwrap_or(0)
    }

    /// Units one page of the widest group occupies.
    fn units_per_page(plan: &CachePlan) -> u32 {
        plan.groups
            .iter()
            .map(|group| group.units_per_page)
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// Take the region of `pages` pages from `pools`, sized by `plan`: both
    /// value planes, the flag plane of each field, and the scale plane of each
    /// on a quantized pool, each over every column of the plan, because a
    /// layer's cell is `column * num_units + unit`.
    pub fn allocate(
        plan: &Arc<CachePlan>,
        pages: u32,
        device: DeviceId,
        pools: &mut Pools,
    ) -> Result<Self> {
        if pages == 0 || plan.columns == 0 {
            return Err(crate::Error::Invalid(
                "a scratch page set needs pages and columns".into(),
            ));
        }
        let units_per_page = Self::units_per_page(plan);
        let num_units = pages
            .checked_mul(units_per_page)
            .ok_or_else(|| crate::Error::Invalid("a scratch page set overflows units".into()))?;
        let cells = u64::from(num_units).max(1) * u64::from(plan.columns);

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

        let _ = device;
        Ok(Self {
            plan: Arc::clone(plan),
            pages,
            units_per_page,
            num_units,
            values,
            initialized,
            scales,
        })
    }

    /// The pages the set holds, over all of a batch's rows.
    pub fn pages(&self) -> u32 {
        self.pages
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
    /// The scratch page set a paired forward writes: `pages` pages taken from
    /// the device's `Scratch` pool.
    pub fn scratch(&self, pages: u32, device: DeviceId, pools: &mut Pools) -> Result<ScratchCache> {
        ScratchCache::allocate(self.plan(), pages, device, pools)
    }

    /// The pages a scratch set of `bytes` holds beside `tables` block tables.
    pub fn scratch_pages_within(&self, bytes: u64, tables: u64) -> u32 {
        u32::try_from(ScratchCache::pages_within(self.plan(), bytes, tables)).unwrap_or(u32::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::CachePlan;

    /// A page spans every column of the plan, because a layer's cell is `column * num_units + unit`:
    /// sizing a set by one column gave forty-eight layers one layer's bytes. The figure is Qwen3-30B-A3B's
    /// cache, a BF16 page of 64 tokens over 48 layers of four KV heads of 128.
    #[test]
    fn a_page_of_the_set_spans_every_column() {
        let plan = (CachePlan::dense(64, 48, 4, 128, ipc::PrefixCacheDtype::BFloat16))
            .validated()
            .expect("a dense plan");
        let page = 48 * (2 * 65_536 + 2);
        let bytes = 1_237_319_680;
        let reserved = 7 * SCRATCH_GRANULARITY;
        assert_eq!(
            ScratchCache::pages_within(&plan, bytes, 1),
            (bytes - reserved) / (page + 4)
        );
        assert_eq!(ScratchCache::pages_within(&plan, reserved, 1), 0);
    }
}
