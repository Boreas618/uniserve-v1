//! The scratch pages a shadow forward writes, and their numbering.
//!
//! A shadow forward computes a served batch a second time with another generation's kernels and
//! must not touch the pages requests are served from, so the keys and values it writes go to
//! scratch pages. The set is one region of the device's scratch pool, sized once and reused by
//! every shadow forward; it is never retained at the Storage Server.
//!
//! The set's pages are one numbering, shared by every attention group and by the rows of a
//! batch: the runner hands the pages to the rows in order, each row the pages of the context it
//! reads, so two rows of one batch never write the same page. The set's page table has a row
//! for every page, row `i` naming the pages `i, i + 1, …` and the set's last page where they
//! run past its end, so a row whose pages begin at page `i` gathers table row `i`. A table row is as wide as the
//! set or, when narrower, the cache's pooled table, which bounds the pages a served row reads.
//! A scratch page is one unit, as an attention page is, so a layer's scratch planes are its
//! column of the set, its index keys included when the cache keeps them.

use std::sync::Arc;

use nn::KvPlanes;
use platform::{Buffer, CopyBatch, DeviceId, PoolKind, SourceOrder, Stream, memset_d8, memset_d32};
use tensor::{DType, Tensor};

use super::layout::{CacheLayout, GroupLayout};
use super::planes::{Planes, UNIT_SCALE};
use crate::memory::lend;
use crate::worker::{Error, Result};

/// The bytes of the scratch pool a set of `pages` pages takes for a cache of `layout` whose
/// pooled tables are `width` pages wide: both value planes, the flags of each field, the scales
/// of each for an FP8 cache and the index keys of a cache that keeps them, over every column,
/// and the set's page table, each allocation rounded up to `granularity`, the extent the
/// device's pools map (`platform::Allocator::granularity`). Zero for a set of no pages.
pub(crate) fn bytes(layout: &CacheLayout, pages: u32, width: u32, granularity: u64) -> u64 {
    if pages == 0 {
        return 0;
    }
    let cells = u64::from(pages) * u64::from(layout.columns.max(1));
    let taken = |bytes: u64| bytes.next_multiple_of(granularity.max(1));

    let mut total = 2 * taken(cells * layout.plane_bytes) + 2 * taken(cells);
    if layout.quantized {
        total += 2 * taken(cells * 4);
    }
    if layout.index_bytes() > 0 {
        total += taken(cells * layout.index_bytes());
    }
    total + taken(u64::from(pages) * u64::from(table_width(pages, width)) * 4)
}

/// The bytes the shadow path copies when a cache write of a paired forward writes one layer of
/// a set of `pages` pages of a cache of `layout` in place (`ops::Twins::in_place`): the layer's
/// key and value planes over the set, their flags, their scales for an FP8 cache and the
/// layer's index keys, each copy from an `ops::twins::ALIGNMENT` boundary. A paired call is one
/// layer's write, and the region holds one call's twins. Zero for a set of no pages.
pub(crate) fn in_place(layout: &CacheLayout, pages: u32) -> u64 {
    let pages = u64::from(pages);
    let copy = |bytes: u64| bytes.next_multiple_of(ops::twins::ALIGNMENT);

    let mut column = 2 * copy(pages * layout.plane_bytes) + 2 * copy(pages);
    if layout.quantized {
        column += 2 * copy(pages * 4);
    }
    if layout.index_bytes() > 0 {
        column += copy(pages * layout.index_bytes());
    }
    column
}

/// The pages a row of a set of `pages` pages' table names, for a cache whose pooled tables are
/// `width` pages wide: the set's pages, or the pooled width when that is smaller.
fn table_width(pages: u32, width: u32) -> u32 {
    pages.min(width).max(1)
}

/// The page table of a set of `pages` pages whose rows are `width` pages wide, row-major: row
/// `i` names the pages `i, i + 1, …`, each past the set's last page naming the last page.
fn table(pages: u32, width: u32) -> Vec<u32> {
    let last = pages.saturating_sub(1);

    (0..pages)
        .flat_map(|first| (0..width).map(move |page| first.saturating_add(page).min(last)))
        .collect()
}

/// The device side of a set: its buffers of the scratch pool.
struct Buffers {
    device: DeviceId,
    values: [Buffer; 2],
    initialized: [Buffer; 2],
    scales: Option<[Buffer; 2]>,
    /// The index-key planes of a cache whose layers keep index keys.
    index_keys: Option<Buffer>,
    /// `[pages, width]` U32: row `i` names the pages from page `i` on ([`table`]).
    table: Buffer,
}

/// The scratch page set of one cache.
pub(crate) struct ScratchPages {
    layout: Arc<CacheLayout>,
    /// Pages the set holds, each one unit of every column.
    pages: u32,
    /// The pages a row of the set's page table names ([`table_width`]).
    width: u32,
    /// The set's buffers; `None` for a set that holds no device.
    device: Option<Buffers>,
}

impl ScratchPages {
    /// A set of `pages` pages that holds no device, for a cache whose pooled tables are `width`
    /// pages wide: its tensors are meta tensors.
    pub(crate) fn meta(layout: Arc<CacheLayout>, pages: u32, width: u32) -> Result<Self> {
        if pages == 0 || layout.columns == 0 {
            return Err(Error::Invalid(
                "a scratch page set needs pages and columns".into(),
            ));
        }

        Ok(Self {
            layout,
            pages,
            width: table_width(pages, width),
            device: None,
        })
    }

    /// A set of `pages` pages taken from the scratch pool of `allocator`, for a cache whose
    /// pooled tables are `width` pages wide ([`bytes`]): both value planes, the flags of each
    /// field, the scales of each for an FP8 cache and the index keys of a cache that keeps
    /// them, over every column, and the page table. The flags and index keys are zeroed, the
    /// scales set to one and the table written on `stream`, which is synchronized before
    /// returning.
    ///
    /// # Errors
    ///
    /// When the pool cannot hold the set.
    pub(crate) fn open(
        layout: Arc<CacheLayout>,
        pages: u32,
        width: u32,
        allocator: &mut platform::Allocator,
        stream: &Stream,
    ) -> Result<Self> {
        let mut set = Self::meta(layout, pages, width)?;
        let cells = pages as usize * set.layout.columns as usize;
        let plane_bytes = cells * set.layout.plane_bytes as usize;
        let index_bytes = cells * set.layout.index_bytes() as usize;

        let mut take = |bytes: usize| allocator.allocate(PoolKind::Scratch, bytes);
        let values = [take(plane_bytes)?, take(plane_bytes)?];
        let initialized = [take(cells)?, take(cells)?];
        let scales = if set.layout.quantized {
            Some([take(cells * 4)?, take(cells * 4)?])
        } else {
            None
        };
        let index_keys = if index_bytes > 0 {
            Some(take(index_bytes)?)
        } else {
            None
        };
        let numbering = table(pages, set.width);
        let table = take(std::mem::size_of_val(numbering.as_slice()))?;

        let mut write = CopyBatch::new(SourceOrder::DuringCall);
        write.push(
            table.address(),
            numbering.as_ptr() as u64,
            std::mem::size_of_val(numbering.as_slice()),
        );

        // SAFETY: every range is one of the buffers just taken, which the set keeps; the host
        // table is read during the call and the stream is drained below.
        unsafe {
            for flags in &initialized {
                memset_d8(flags.address(), 0, cells, stream)?;
            }
            for scales in scales.iter().flatten() {
                memset_d32(scales.address(), UNIT_SCALE, cells, stream)?;
            }

            // The index keys have no flag: zeroed once, they hold finite keys whatever a
            // forward reads of them.
            if let Some(index_keys) = &index_keys {
                memset_d8(index_keys.address(), 0, index_bytes, stream)?;
            }
            write.submit(stream)?;
        }
        stream.synchronize()?;

        set.device = Some(Buffers {
            device: allocator.device(),
            values,
            initialized,
            scales,
            index_keys,
            table,
        });
        Ok(set)
    }

    /// The pages the set holds, over all of a batch's rows.
    pub(crate) fn pages(&self) -> u32 {
        self.pages
    }

    /// The tensor of `shape` elements of `dtype` `offset` bytes into `buffer` of the device
    /// side, or the meta tensor of that shape.
    fn tensor(
        &self,
        buffer: impl FnOnce(&Buffers) -> &Buffer,
        offset: u64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        match &self.device {
            // SAFETY: the set owns the buffer for as long as it lives, the callers bound the
            // window to the set's cells, and only the shadow forwards issued on the tensors
            // and the copies this set issues address it.
            Some(buffers) => unsafe {
                lend(
                    buffer(buffers).address() + offset,
                    shape,
                    dtype,
                    Some(buffers.device),
                )
            },
            // SAFETY: a meta tensor has no data and its address is never read.
            None => unsafe { lend(0, shape, dtype, None) },
        }
    }

    /// Layer `layer` of the attention group `group` over the scratch pages: the shape the
    /// cache's own planes have, with this set's pages.
    pub(crate) fn layer(&self, group: &GroupLayout, layer: usize) -> Result<KvPlanes> {
        if layer >= group.layers.len() || layer as u32 >= self.layout.columns {
            return Err(Error::Invalid(format!(
                "a cache group of {} layers holds no layer {layer}",
                group.layers.len()
            )));
        }
        if u64::from(group.block_size) * group.row_bytes() != self.layout.plane_bytes {
            return Err(Error::Invariant(format!(
                "a page of {} tokens of {} bytes does not fill a plane of {} bytes",
                group.block_size,
                group.row_bytes(),
                self.layout.plane_bytes
            )));
        }

        // The layer's column: its cells are the set's pages in order.
        let cell = layer as u64 * u64::from(self.pages);
        let pages = self.pages as usize;
        let values = [
            pages,
            group.block_size as usize,
            group.num_kv_heads as usize,
            group.head_dim as usize,
        ];
        let cells = [pages, 1, 1, 1];
        let plane_bytes = self.layout.plane_bytes;

        let value = |field: usize| {
            self.tensor(
                |buffers| &buffers.values[field],
                cell * plane_bytes,
                &values,
                group.dtype,
            )
        };
        let flags = |field: usize| {
            self.tensor(
                |buffers| &buffers.initialized[field],
                cell,
                &cells,
                DType::U8,
            )
        };
        let scale = |field: usize| -> Result<Option<Tensor>> {
            if !self.layout.quantized {
                return Ok(None);
            }
            let scales = match &self.device {
                Some(buffers) => {
                    let scales = buffers.scales.as_ref().ok_or_else(|| {
                        Error::Invariant("an FP8 scratch set holds no scale buffer".into())
                    })?;

                    // SAFETY: as in `Self::tensor`.
                    unsafe {
                        lend(
                            scales[field].address() + cell * 4,
                            &cells,
                            DType::F32,
                            Some(buffers.device),
                        )
                    }?
                }
                // SAFETY: a meta tensor has no data and its address is never read.
                None => unsafe { lend(0, &cells, DType::F32, None) }?,
            };
            Ok(Some(scales))
        };

        // The layer's index keys are the same cells of the index-key buffer.
        let index_keys = || -> Result<Option<Tensor>> {
            let Some(shape) = self.layout.index_shape(group, pages, group.block_size)? else {
                return Ok(None);
            };
            let keys = match &self.device {
                Some(buffers) => {
                    let index_keys = buffers.index_keys.as_ref().ok_or_else(|| {
                        Error::Invariant(
                            "a scratch set of a cache that keeps index keys holds no buffer of \
                             them"
                                .into(),
                        )
                    })?;

                    // SAFETY: as in `Self::tensor`; a page's keys fill one index-key plane, so
                    // the window is the buffer's cells from `cell`.
                    unsafe {
                        lend(
                            index_keys.address() + cell * self.layout.index_bytes(),
                            &shape,
                            group.compute_dtype,
                            Some(buffers.device),
                        )
                    }?
                }
                // SAFETY: a meta tensor has no data and its address is never read.
                None => unsafe { lend(0, &shape, group.compute_dtype, None) }?,
            };
            Ok(Some(keys))
        };

        Ok(KvPlanes {
            key: value(0)?,
            value: value(1)?,
            key_initialized: flags(0)?,
            value_initialized: flags(1)?,
            key_scale: scale(0)?,
            value_scale: scale(1)?,
            index_keys: index_keys()?,
        })
    }

    /// The set's page table, `[pages, width]` U32: row `i` names the pages from page `i` on, so
    /// a row of a shadow forward whose pages begin at page `i` gathers row `i`.
    ///
    /// # Errors
    ///
    /// As [`lend`](crate::memory::lend).
    pub(crate) fn table(&self) -> Result<Tensor> {
        self.tensor(
            |buffers| &buffers.table,
            0,
            &[self.pages as usize, self.width as usize],
            DType::U32,
        )
    }

    /// Copies what the rows of a batch have cached in attention group `group` into their
    /// scratch pages on `stream`: for each row, `(first, units)`, the units of its logical
    /// pages from page zero (unit 0 where it holds none) into the set's pages from `first` on,
    /// every layer's keys, values, flags, scales and index keys. The shadow forward then reads
    /// the row's whole history in the set and writes over the copy.
    ///
    /// # Errors
    ///
    /// A set that holds no device, or a row whose pages lie past the set's.
    pub(crate) fn copy_history(
        &self,
        planes: &Planes,
        group: &GroupLayout,
        rows: &[(u32, Vec<u32>)],
        stream: &Stream,
    ) -> Result<()> {
        let Some(buffers) = &self.device else {
            return Err(Error::State(
                "the scratch page set holds no device: it copies no history",
            ));
        };
        let past = rows
            .iter()
            .find(|(first, units)| u64::from(*first) + units.len() as u64 > u64::from(self.pages));
        if let Some((first, units)) = past {
            return Err(Error::Invalid(format!(
                "a shadow forward copies a history of {} pages from page {first} of a scratch \
                 set of {} pages",
                units.len(),
                self.pages
            )));
        }

        let plane_bytes = self.layout.plane_bytes;
        let index_bytes = self.layout.index_bytes();
        let mut batch = CopyBatch::new(SourceOrder::Stream);

        for (first, units) in rows {
            for (page, unit) in units.iter().enumerate() {
                if *unit == 0 {
                    continue;
                }
                let target = u64::from(*first) + page as u64;

                for layer in 0..group.layers.len() as u32 {
                    let cell = u64::from(layer) * u64::from(self.pages) + target;

                    for field in 0..2 {
                        batch.push(
                            buffers.values[field].address() + cell * plane_bytes,
                            planes.plane_address(field, layer, *unit)?,
                            plane_bytes as usize,
                        );
                        batch.push(
                            buffers.initialized[field].address() + cell,
                            planes.flag_address(field, layer, *unit)?,
                            1,
                        );
                        if let Some(scales) = &buffers.scales {
                            batch.push(
                                scales[field].address() + cell * 4,
                                planes.scale_address(field, layer, *unit)?,
                                4,
                            );
                        }
                    }

                    if let Some(index_keys) = &buffers.index_keys {
                        batch.push(
                            index_keys.address() + cell * index_bytes,
                            planes.index_address(layer, *unit)?,
                            index_bytes as usize,
                        );
                    }
                }
            }
        }

        // SAFETY: the destinations are distinct cells of the set's buffers, bounded above, and
        // the sources cells of the cache's planes, which the requests of the served batch hold
        // until it completes; the shadow forward follows on this stream.
        unsafe { batch.submit(stream) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nn::family::IndexKeys;
    use proto::request::PrefixCacheDtype;

    use super::*;

    /// A page of the set spans every column of the cache, a layer's cell being one column of
    /// one page, and every allocation takes whole granules: the figure is a set of 384 BF16
    /// pages of 64 tokens over 48 layers of four heads of 128, in a cache whose pooled tables
    /// are 640 pages wide, and the table's rows the set's 384 pages wide.
    #[test]
    fn a_set_takes_its_pages_over_every_column_and_its_table() {
        let layout = CacheLayout::dense(64, 48, 4, 128, PrefixCacheDtype::BFloat16);
        let cells = 384 * 48;
        // The extent the pools of a device map in, which a test states.
        let granularity = 2 << 20;
        let granules = |bytes: u64| bytes.next_multiple_of(granularity);

        let values = 2 * granules(cells * 65_536);
        let flags = 2 * granules(cells);
        let table = granules(384 * 384 * 4);
        assert_eq!(
            bytes(&layout, 384, 640, granularity),
            values + flags + table
        );
        assert_eq!(bytes(&layout, 0, 640, granularity), 0);

        // Index keys of 128 elements per four tokens add sixteen BF16 keys to every cell, in
        // an allocation of their own; a pooled table narrower than the set narrows its table.
        let keyed = layout
            .with_index_keys(IndexKeys {
                width: 128,
                ratio: 4,
            })
            .expect("index keys a page holds");
        let keys = granules(cells * 16 * 128 * 2);
        let table = granules(384 * 256 * 4);
        assert_eq!(
            bytes(&keyed, 384, 256, granularity),
            values + flags + keys + table
        );
    }

    /// A set's planes have the cache's page shape and element type over the set's pages, so a
    /// paired forward writes and reads them as the serving forward does the cache's own (an FP16
    /// cache of a BF16 model through the same converting write), and its table reads at any
    /// width the set's pages cover.
    #[test]
    fn a_sets_tensors_are_its_pages_in_the_caches_shape() -> Result<()> {
        let float16 = Arc::new(CacheLayout::dense(16, 2, 1, 8, PrefixCacheDtype::Float16));
        let layer =
            ScratchPages::meta(Arc::clone(&float16), 12, 4)?.layer(&float16.groups[0], 0)?;
        assert_eq!(
            (layer.key.dtype(), layer.value.dtype()),
            (DType::F16, DType::F16)
        );
        assert!(layer.key_scale.is_none() && layer.value_scale.is_none());

        let layout = Arc::new(CacheLayout::dense(
            16,
            2,
            1,
            8,
            PrefixCacheDtype::Float8E4m3Fn,
        ));
        let set = ScratchPages::meta(Arc::clone(&layout), 12, 4)?;
        assert_eq!(set.pages(), 12);

        let layer = set.layer(&layout.groups[0], 1)?;
        assert_eq!(layer.key.dims(), [12, 16, 1, 8]);
        assert_eq!(layer.key.dtype(), DType::F8E4M3);
        assert_eq!(layer.value_initialized.dims(), [12, 1, 1, 1]);
        assert!(layer.key_scale.is_some());
        assert!(layer.index_keys.is_none());
        assert!(set.layer(&layout.groups[0], 2).is_err());

        // A row for every page, as wide as the pooled tables, which are narrower than the set.
        let table = set.table()?;
        assert_eq!((table.dims(), table.dtype()), (&[12, 4][..], DType::U32));
        assert_eq!(
            ScratchPages::meta(Arc::clone(&layout), 3, 4)?
                .table()?
                .dims(),
            [3, 3]
        );
        assert!(ScratchPages::meta(layout, 0, 4).is_err());
        Ok(())
    }

    /// A cache write of a paired forward copies one layer's planes and flags over the set: the
    /// figure is the set of 384 BF16 pages of 64 tokens of a cache of 48 layers of four heads of
    /// 128, whatever its layer count.
    #[test]
    fn a_cache_write_of_a_paired_forward_copies_one_layer_of_the_set() {
        let layout = CacheLayout::dense(64, 48, 4, 128, PrefixCacheDtype::BFloat16);

        // The layer's two planes of 384 pages of 64 KiB, and its two flag planes of 384 bytes,
        // each copy taking whole blocks of 256 bytes.
        assert_eq!(in_place(&layout, 384), 2 * 384 * 65_536 + 2 * 512);
        assert_eq!(
            in_place(
                &CacheLayout::dense(64, 2, 4, 128, PrefixCacheDtype::BFloat16),
                384
            ),
            in_place(&layout, 384)
        );
        assert_eq!(in_place(&layout, 0), 0);
    }

    /// Row `i` of the set's table names the pages from page `i` on, and a page past the set's
    /// last names the last: a row whose pages begin at page `i` reads them from column zero.
    #[test]
    fn a_table_row_names_the_pages_from_its_own_on() {
        assert_eq!(
            table(4, 3),
            [
                0, 1, 2, //
                1, 2, 3, //
                2, 3, 3, //
                3, 3, 3,
            ]
        );
    }

    /// A set of a cache that keeps index keys lends them over its own pages, in the shape the
    /// cache's planes lend them.
    #[test]
    fn a_set_lends_the_index_keys_of_its_pages() -> Result<()> {
        let keys = IndexKeys { width: 6, ratio: 4 };
        let layout = Arc::new(
            CacheLayout::dense(16, 2, 1, 8, PrefixCacheDtype::BFloat16).with_index_keys(keys)?,
        );
        let set = ScratchPages::meta(Arc::clone(&layout), 12, 4)?;

        let layer = set.layer(&layout.groups[0], 1)?;
        let index_keys = layer.index_keys.expect("the layer keeps index keys");
        assert_eq!(
            (index_keys.dims(), index_keys.dtype()),
            (&[12, 4, 6][..], DType::BF16)
        );
        Ok(())
    }
}
