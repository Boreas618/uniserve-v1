//! The scratch pages a shadow forward writes, and their numbering.
//!
//! A shadow forward computes a served batch a second time with another generation's kernels and
//! must not touch the pages requests are served from, so the entries it writes go to scratch
//! pages. The set is one region of the device's scratch pool, sized once and reused by every
//! shadow forward; it is never retained at the Storage Server.
//!
//! The set's pages are one numbering, shared by every field group of the KV pool and by the rows
//! of a batch: the runner hands the pages to the rows in order, each row the pages of the context
//! it reads, so two rows of one batch never write the same page. The set's page table has a row
//! for every page, row `i` naming the pages `i, i + 1, …` and the set's last page where they
//! run past its end, so a row whose pages begin at page `i` gathers table row `i`. A table row is
//! as wide as the set or, when narrower, the cache's pooled table, which bounds the pages a
//! served row reads. A scratch page is one unit of the KV pool, as a full-history page is, so an
//! owner's scratch plane of a field is its column of the set. A cache with a sliding-window group
//! computes no paired forward, so no set holds a window's pages.

use std::sync::Arc;

use nn::KvPlane;
use platform::{Buffer, CopyBatch, DeviceId, PoolKind, SourceOrder, Stream, memset_d8, memset_d32};
use tensor::{DType, Tensor};

use super::layout::{CacheLayout, FieldLayout, GroupLayout};
use super::planes::{Planes, UNIT_SCALE};
use crate::memory::lend;
use crate::worker::{Error, Result};

/// The bytes of the scratch pool a set of `pages` pages takes for a cache of `layout` whose
/// pooled tables are `width` pages wide: every field's planes, flags and, for a scaled field,
/// scales over every column, and the set's page table, each allocation rounded up to
/// `granularity`, the extent the device's pools map (`platform::Allocator::granularity`). Zero
/// for a set of no pages.
pub(crate) fn bytes(layout: &CacheLayout, pages: u32, width: u32, granularity: u64) -> u64 {
    if pages == 0 {
        return 0;
    }
    let pool = layout.kv();
    let taken = |bytes: u64| bytes.next_multiple_of(granularity.max(1));
    let pages_bytes = u64::from(pages);

    let fields: u64 = pool
        .fields
        .iter()
        .map(|field| {
            let cells = pages_bytes * u64::from(field.columns());
            let mut total = taken(pages_bytes * field.unit_bytes(pool.block_size)) + taken(cells);
            if field.scaled {
                total += taken(cells * 4);
            }
            total
        })
        .sum();
    fields + taken(pages_bytes * u64::from(table_width(pages, width)) * 4)
}

/// The bytes the shadow path copies when a cache write of a paired forward writes one layer of
/// a set of `pages` pages of a cache of `layout` in place (`ops::Twins::in_place`): the layer's
/// plane of every field over the set (the widest column's), its flags and its scales for a scaled
/// field, each copy from an `ops::allocator::ALIGNMENT` boundary. A paired call is one layer's
/// write, and the region holds one call's twins. Zero for a set of no pages.
pub(crate) fn in_place(layout: &CacheLayout, pages: u32) -> u64 {
    let pool = layout.kv();
    let pages = u64::from(pages);
    let copy = |bytes: u64| bytes.next_multiple_of(ops::allocator::ALIGNMENT);

    pool.fields
        .iter()
        .map(|field| {
            let plane = (0..field.owners.len())
                .map(|column| field.plane_bytes(column, pool.block_size))
                .max()
                .unwrap_or(0);
            let mut column = copy(pages * plane) + copy(pages);
            if field.scaled {
                column += copy(pages * 4);
            }
            column
        })
        .sum()
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

/// The buffers of one field of a set: its planes, column-major, its flags and its scales.
struct FieldBuffers {
    values: Buffer,
    initialized: Buffer,
    scale: Option<Buffer>,
}

/// The device side of a set: its buffers of the scratch pool.
struct Buffers {
    device: DeviceId,
    /// One per field of the KV pool, in its order.
    fields: Vec<FieldBuffers>,
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
        if pages == 0 || layout.columns() == 0 {
            return Err(Error::Invalid(
                "a scratch page set needs pages and columns".into(),
            ));
        }
        if let Some((group, _)) = layout.window_groups().first() {
            return Err(Error::Unsupported(format!(
                "a cache with the sliding-window group {group} computes no paired forward, so it \
                 keeps no scratch pages"
            )));
        }

        Ok(Self {
            layout,
            pages,
            width: table_width(pages, width),
            device: None,
        })
    }

    /// A set of `pages` pages taken from the scratch pool of `allocator`, for a cache whose
    /// pooled tables are `width` pages wide ([`bytes`]): every field's planes, flags and, for a
    /// scaled field, scales over every column, and the page table. The flags are zeroed, the
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
        let block = set.layout.kv().block_size;

        let mut take = |bytes: usize| allocator.allocate(PoolKind::Scratch, bytes);
        let mut fields = Vec::new();
        for field in &set.layout.kv().fields {
            let cells = pages as usize * field.columns() as usize;
            fields.push(FieldBuffers {
                values: take(pages as usize * field.unit_bytes(block) as usize)?,
                initialized: take(cells)?,
                scale: field.scaled.then(|| take(cells * 4)).transpose()?,
            });
        }
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
            for (buffers, field) in fields.iter().zip(&set.layout.kv().fields) {
                let cells = pages as usize * field.columns() as usize;
                memset_d8(buffers.initialized.address(), 0, cells, stream)?;
                if let Some(scale) = &buffers.scale {
                    memset_d32(scale.address(), UNIT_SCALE, cells, stream)?;
                }
            }
            write.submit(stream)?;
        }
        stream.synchronize()?;

        set.device = Some(Buffers {
            device: allocator.device(),
            fields,
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
        buffer: impl FnOnce(&Buffers) -> Option<&Buffer>,
        offset: u64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        match &self.device {
            Some(buffers) => {
                let buffer = buffer(buffers).ok_or_else(|| {
                    Error::Invariant("a scratch set holds no buffer of a field it lends".into())
                })?;
                // SAFETY: the set owns the buffer for as long as it lives, the callers bound the
                // window to the set's cells, and only the shadow forwards issued on the tensors
                // and the copies this set issues address it.
                unsafe {
                    lend(
                        buffer.address() + offset,
                        shape,
                        dtype,
                        Some(buffers.device),
                    )
                }
            }
            // SAFETY: a meta tensor has no data and its address is never read.
            None => unsafe { lend(0, shape, dtype, None) },
        }
    }

    /// The bytes before column `column`'s first page in the planes of `field` over the set.
    fn column_offset(&self, field: &FieldLayout, column: usize) -> u64 {
        field.column_offset(column, self.pages, self.layout.kv().block_size)
    }

    /// The planes of the field group `group` of the KV pool over the scratch pages: the shapes
    /// the cache's own planes have ([`Planes::planes`]), with this set's pages.
    pub(crate) fn planes(&self, group: &GroupLayout) -> Result<Vec<KvPlane>> {
        if group.pool != 0 || !group.attention() {
            return Err(Error::Invalid(format!(
                "a scratch set holds pages of the KV pool's field group, not of group {:?}",
                group.layers
            )));
        }

        let pages = self.pages as usize;
        let block = self.layout.kv().block_size;
        let mut planes = Vec::new();
        for (index, field) in self.layout.kv().fields.iter().enumerate() {
            let [heads, width] = field.dims;
            for (column, owner) in field.owners.iter().enumerate() {
                let entries = field.entries(column, block) as usize;
                let values = [pages, entries, heads as usize, width as usize];
                let cells = [pages, 1, 1, 1];
                let cell = column as u64 * u64::from(self.pages);

                planes.push(KvPlane {
                    field: field.name,
                    layer: owner.layer,
                    readers: field
                        .readers
                        .iter()
                        .filter(|reader| reader.owner == owner.layer)
                        .map(|reader| reader.layer)
                        .collect(),
                    ratio: owner.ratio,
                    entries: self.tensor(
                        |buffers| Some(&buffers.fields[index].values),
                        self.column_offset(field, column),
                        &values,
                        field.dtype,
                    )?,
                    initialized: self.tensor(
                        |buffers| Some(&buffers.fields[index].initialized),
                        cell,
                        &cells,
                        DType::U8,
                    )?,
                    scale: field
                        .scaled
                        .then(|| {
                            self.tensor(
                                |buffers| buffers.fields[index].scale.as_ref(),
                                cell * 4,
                                &cells,
                                DType::F32,
                            )
                        })
                        .transpose()?,
                });
            }
        }
        Ok(planes)
    }

    /// The set's page table, `[pages, width]` U32: row `i` names the pages from page `i` on, so
    /// a row of a shadow forward whose pages begin at page `i` gathers row `i`.
    ///
    /// # Errors
    ///
    /// As [`lend`](crate::memory::lend).
    pub(crate) fn table(&self) -> Result<Tensor> {
        self.tensor(
            |buffers| Some(&buffers.table),
            0,
            &[self.pages as usize, self.width as usize],
            DType::U32,
        )
    }

    /// Copies what the rows of a batch have cached in the KV pool's field group into their
    /// scratch pages on `stream`: for each row, `(first, units)`, the units of its logical pages
    /// from page zero (unit 0 where it holds none) into the set's pages from `first` on, every
    /// owner's planes of every field with their flags and scales. The shadow forward then reads
    /// the row's whole history in the set and writes over the copy.
    ///
    /// # Errors
    ///
    /// A set that holds no device, or a row whose pages lie past the set's.
    pub(crate) fn copy_history(
        &self,
        planes: &Planes,
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

        let block = self.layout.kv().block_size;
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for (first, units) in rows {
            for (page, unit) in units.iter().enumerate() {
                if *unit == 0 {
                    continue;
                }
                let target = u64::from(*first) + page as u64;

                for (index, field) in self.layout.kv().fields.iter().enumerate() {
                    let held = &buffers.fields[index];
                    for column in 0..field.columns() {
                        let plane = field.plane_bytes(column as usize, block);
                        let cell = u64::from(column) * u64::from(self.pages) + target;

                        batch.push(
                            held.values.address()
                                + self.column_offset(field, column as usize)
                                + target * plane,
                            planes.plane_address(index, column, *unit)?,
                            plane as usize,
                        );
                        batch.push(
                            held.initialized.address() + cell,
                            planes.flag_address(index, column, *unit)?,
                            1,
                        );
                        if let Some(scale) = &held.scale {
                            batch.push(
                                scale.address() + cell * 4,
                                planes.scale_address(index, column, *unit)?,
                                4,
                            );
                        }
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
    use nn::family::{CacheField, FieldOwner, Format, History, INDEX_KEY, KEY, Record, VALUE};
    use proto::request::PrefixCacheDtype;

    use super::*;

    /// `layers` layers of `heads` heads of 128 over `block`-token pages, in `dtype`.
    fn dense(block: u32, layers: u32, heads: u32, dtype: PrefixCacheDtype) -> CacheLayout {
        CacheLayout::of(
            block,
            &CacheField::keys_and_values(layers, heads, 128),
            dtype,
        )
        .expect("a dense layout")
    }

    /// Index keys of `width` BF16 elements per four tokens over `layers` layers.
    fn index_keys(layers: u32, width: u32) -> CacheField {
        CacheField {
            name: INDEX_KEY,
            owners: (0..layers)
                .map(|layer| FieldOwner { layer, ratio: 4 })
                .collect(),
            readers: Vec::new(),
            history: History::Full,
            record: Record {
                format: Format::Elements(DType::BF16),
                heads: 1,
                width,
            },
        }
    }

    /// A page of the set spans every column of the cache, an owner's cell being one column of
    /// one page, and every allocation takes whole granules: the figure is a set of 384 BF16
    /// pages of 64 tokens over 48 layers of four heads of 128, in a cache whose pooled tables
    /// are 640 pages wide, and the table's rows the set's 384 pages wide.
    #[test]
    fn a_set_takes_its_pages_over_every_column_and_its_table() {
        let layout = dense(64, 48, 4, PrefixCacheDtype::BFloat16);
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

        // Index keys of 128 elements per four tokens add sixteen BF16 keys and a flag to every
        // cell, in allocations of their own; a pooled table narrower than the set narrows its
        // table.
        let mut fields = CacheField::keys_and_values(48, 4, 128);
        fields.push(index_keys(48, 128));
        let keyed = CacheLayout::of(64, &fields, PrefixCacheDtype::BFloat16)
            .expect("index keys a page holds");
        let keys = granules(cells * 16 * 128 * 2) + granules(cells);
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
        let float16 = Arc::new(CacheLayout::of(
            16,
            &CacheField::keys_and_values(2, 1, 8),
            PrefixCacheDtype::Float16,
        )?);
        let planes = ScratchPages::meta(Arc::clone(&float16), 12, 4)?.planes(&float16.groups[0])?;
        assert!(
            planes
                .iter()
                .all(|plane| plane.entries.dtype() == DType::F16)
        );
        assert!(planes.iter().all(|plane| plane.scale.is_none()));

        let layout = Arc::new(CacheLayout::of(
            16,
            &CacheField::keys_and_values(2, 1, 8),
            PrefixCacheDtype::Float8E4m3Fn,
        )?);
        let set = ScratchPages::meta(Arc::clone(&layout), 12, 4)?;
        assert_eq!(set.pages(), 12);

        let planes = set.planes(&layout.groups[0])?;
        let fields: Vec<(&str, u32)> = planes
            .iter()
            .map(|plane| (plane.field, plane.layer))
            .collect();
        assert_eq!(fields, [(KEY, 0), (KEY, 1), (VALUE, 0), (VALUE, 1)]);
        assert_eq!(planes[1].entries.dims(), [12, 16, 1, 8]);
        assert_eq!(planes[1].entries.dtype(), DType::F8E4M3);
        assert_eq!(planes[3].initialized.dims(), [12, 1, 1, 1]);
        assert!(planes[1].scale.is_some());

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
        let layout = dense(64, 48, 4, PrefixCacheDtype::BFloat16);

        // The layer's two planes of 384 pages of 64 KiB, and its two flag planes of 384 bytes,
        // each copy taking whole blocks of 256 bytes.
        assert_eq!(in_place(&layout, 384), 2 * 384 * 65_536 + 2 * 512);
        assert_eq!(
            in_place(&dense(64, 2, 4, PrefixCacheDtype::BFloat16), 384),
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
        let mut fields = CacheField::keys_and_values(2, 1, 8);
        fields.push(index_keys(2, 6));
        let layout = Arc::new(CacheLayout::of(16, &fields, PrefixCacheDtype::BFloat16)?);
        let set = ScratchPages::meta(Arc::clone(&layout), 12, 4)?;

        let planes = set.planes(&layout.groups[0])?;
        let index = planes
            .iter()
            .find(|plane| plane.field == INDEX_KEY && plane.layer == 1)
            .expect("the layer keeps index keys");
        assert_eq!(
            (index.entries.dims(), index.entries.dtype()),
            (&[12, 4, 1, 6][..], DType::BF16)
        );
        Ok(())
    }
}
