//! The scratch pages a shadow forward writes, and their numbering.
//!
//! A shadow forward computes a served batch a second time with another generation's kernels and
//! must not touch the pages requests are served from, so the keys and values it writes go to
//! scratch pages. The set is one region of the device's scratch pool, sized once and reused by
//! every shadow forward; it is never retained at the Storage Server.
//!
//! The set's pages are one numbering, shared by every attention group. A batch whose rows need
//! `pages` pages each gives row `r` the pages `r · pages .. (r + 1) · pages`, so two rows of one
//! batch never write the same page, and the set's page table, which states the numbering once
//! in order, reads as `[rows, pages]` at any width. A scratch page is one unit, as an
//! attention page is, so a layer's scratch planes are its column of the set.

use std::sync::Arc;

use nn::{KvPlanes, NO_SLOT};
use platform::{Buffer, CopyBatch, DeviceId, PoolKind, SourceOrder, Stream, memset_d8, memset_d32};
use uniserve_tensor::{DType, Tensor};

use super::layout::{CacheLayout, GroupLayout};
use super::planes::Planes;
use crate::memory::lend;
use crate::{Error, Result};

/// The smallest extent the device's pools map, which every allocation of a set rounds up to.
const SCRATCH_GRANULARITY: u64 = 2 << 20;

/// The allocations a set takes beside its page table: both fields' values, flags and scales.
const ALLOCATIONS: u64 = 6;

/// The bit pattern of `1.0f32`, the neutral FP8 scale.
const UNIT_SCALE: u32 = 0x3F80_0000;

/// The pages `bytes` of the scratch pool hold of a cache of `layout`: both value planes, the
/// flags of each field, and the scales of each for an FP8 cache, over every column, and the
/// page's entry of the set's page table, after the granularity each of the set's allocations
/// rounds up to.
pub fn pages_within(layout: &CacheLayout, bytes: u64) -> u64 {
    let scales = if layout.quantized { 2 * 4 } else { 0 };
    let page = u64::from(layout.columns.max(1)) * (2 * layout.plane_bytes + 2 + scales);

    bytes
        .saturating_sub((ALLOCATIONS + 1) * SCRATCH_GRANULARITY)
        .checked_div(page + 4)
        .unwrap_or(0)
}

/// The device side of a set: its buffers of the scratch pool.
struct Buffers {
    device: DeviceId,
    values: [Buffer; 2],
    initialized: [Buffer; 2],
    scales: Option<[Buffer; 2]>,
    /// `[pages]` U32: the page numbers `0..pages` in order.
    table: Buffer,
}

/// The scratch page set of one cache.
pub(crate) struct ScratchPages {
    layout: Arc<CacheLayout>,
    /// Pages the set holds, each one unit of every column.
    pages: u32,
    /// The set's buffers; `None` for a set that holds no device.
    device: Option<Buffers>,
}

impl ScratchPages {
    /// A set of `pages` pages that holds no device: its tensors are meta tensors.
    pub(crate) fn meta(layout: Arc<CacheLayout>, pages: u32) -> Result<Self> {
        if pages == 0 || layout.columns == 0 {
            return Err(Error::Invalid(
                "a scratch page set needs pages and columns".into(),
            ));
        }

        Ok(Self {
            layout,
            pages,
            device: None,
        })
    }

    /// A set of `pages` pages taken from the scratch pool of `allocator`: both value planes,
    /// the flags of each field and the scales of each for an FP8 cache, over every column, and
    /// the page table. The flags are zeroed, the scales set to one and the table written on
    /// `stream`, which is synchronized before returning.
    ///
    /// # Errors
    ///
    /// When the pool cannot hold the set.
    pub(crate) fn open(
        layout: Arc<CacheLayout>,
        pages: u32,
        allocator: &mut platform::Allocator,
        stream: &Stream,
    ) -> Result<Self> {
        let mut set = Self::meta(layout, pages)?;
        let cells = pages as usize * set.layout.columns as usize;
        let plane_bytes = cells * set.layout.plane_bytes as usize;

        let mut take = |bytes: usize| allocator.allocate(PoolKind::Scratch, bytes);
        let values = [take(plane_bytes)?, take(plane_bytes)?];
        let initialized = [take(cells)?, take(cells)?];
        let scales = if set.layout.quantized {
            Some([take(cells * 4)?, take(cells * 4)?])
        } else {
            None
        };
        let table = take(pages as usize * 4)?;

        let numbering: Vec<u32> = (0..pages).collect();
        let mut write = CopyBatch::new(SourceOrder::DuringCall);
        write.push(
            table.address(),
            numbering.as_ptr() as u64,
            std::mem::size_of_val(numbering.as_slice()),
        );

        // SAFETY: every range is one of the buffers just taken, which the set keeps; the host
        // numbering is read during the call and the stream is drained below.
        unsafe {
            for flags in &initialized {
                memset_d8(flags.address(), 0, cells, stream)?;
            }
            for scales in scales.iter().flatten() {
                memset_d32(scales.address(), UNIT_SCALE, cells, stream)?;
            }
            write.submit(stream)?;
        }
        stream.synchronize()?;

        set.device = Some(Buffers {
            device: allocator.device(),
            values,
            initialized,
            scales,
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

        Ok(KvPlanes {
            key: value(0)?,
            value: value(1)?,
            key_initialized: flags(0)?,
            value_initialized: flags(1)?,
            key_scale: scale(0)?,
            value_scale: scale(1)?,
        })
    }

    /// The set's page table for a batch of `rows` rows of `pages` pages each, `[rows, pages]`
    /// U32: row `r` names the pages `r · pages .. (r + 1) · pages`.
    ///
    /// # Errors
    ///
    /// When the rows need more pages than the set holds: the forward would write outside its
    /// own pages.
    pub(crate) fn table(&self, rows: u32, pages: u32) -> Result<Tensor> {
        let asked = u64::from(rows) * u64::from(pages);
        if rows == 0 || pages == 0 || asked > u64::from(self.pages) {
            return Err(Error::Invalid(format!(
                "a shadow forward asks for {rows} rows of {pages} pages and the scratch set \
                 holds {} pages",
                self.pages
            )));
        }

        self.tensor(
            |buffers| &buffers.table,
            0,
            &[rows as usize, pages as usize],
            DType::U32,
        )
    }

    /// Copies what the rows of a batch have cached in attention group `group` into their
    /// scratch pages on `stream`: for row `r`, each page of `pages[r]` (the units of the row's
    /// logical pages from page zero, unit 0 where it holds none) into scratch page
    /// `r · width + page`, every layer's keys, values, flags and scales. The shadow forward
    /// then reads the row's whole history in the set and writes over the copy.
    pub(crate) fn copy_history(
        &self,
        planes: &Planes,
        group: &GroupLayout,
        pages: &[Vec<u32>],
        width: u32,
        stream: &Stream,
    ) -> Result<()> {
        let Some(buffers) = &self.device else {
            return Err(Error::State(
                "the scratch page set holds no device: it copies no history",
            ));
        };
        if pages.len() as u64 * u64::from(width) > u64::from(self.pages)
            || pages.iter().any(|row| row.len() > width as usize)
        {
            return Err(Error::Invalid(format!(
                "a shadow forward of {} rows of {width} pages copies its history into a scratch \
                 set of {} pages",
                pages.len(),
                self.pages
            )));
        }

        let plane_bytes = self.layout.plane_bytes;
        let mut batch = CopyBatch::new(SourceOrder::Stream);

        for (row, units) in pages.iter().enumerate() {
            for (page, unit) in units.iter().enumerate() {
                if *unit == 0 {
                    continue;
                }
                let target = row as u64 * u64::from(width) + page as u64;

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

/// The cache slots a shadow forward writes: the tokens of the served batch, in the scratch
/// set's numbering.
///
/// A slot is flat over one plane's pages (`page · page_size + offset`), and the scratch set
/// gives row `r` the pages `r · pages .. (r + 1) · pages`. So token `t` of row `r` at position
/// `p` writes `(r · pages + p / page_size) · page_size + p % page_size`, which is inside the
/// row's own pages and never a slot a request is served from. A token the served batch does not
/// write (`slots[t] == NO_SLOT`, a padding token) writes nothing here either.
///
/// `positions` is the batch's `[tokens]` rotary positions, `query_offsets` its `[rows + 1]`
/// packed offsets, and `slots` its `[tokens]` slot column. The result is the `[tokens]` slot
/// column the shadow forward reads in place of the served batch's, [`NO_SLOT`] where a token
/// writes nothing.
///
/// # Errors
///
/// Columns that describe different token counts, a negative position, or a position past the
/// row's `pages` pages.
pub fn scratch_slot_mapping(
    positions: &[i32],
    query_offsets: &[u32],
    slots: &[u32],
    pages: u32,
    page_size: u32,
) -> Result<Vec<u32>> {
    if page_size == 0 || pages == 0 {
        return Err(Error::Invariant(
            "a scratch page set needs pages of at least one token".into(),
        ));
    }
    if query_offsets.len() < 2 {
        return Err(Error::Invariant(
            "a batch's packed query offsets name no row".into(),
        ));
    }
    match query_offsets.last() {
        Some(tokens) if *tokens as usize == positions.len() && positions.len() == slots.len() => {}
        _ => {
            return Err(Error::Invariant(
                "the batch's offsets, positions and slots describe different token counts".into(),
            ));
        }
    }

    let mut mapping = vec![NO_SLOT; positions.len()];
    for (row, bounds) in query_offsets.windows(2).enumerate() {
        let (start, end) = (bounds[0] as usize, bounds[1] as usize);
        if end < start || end > positions.len() {
            return Err(Error::Invariant(format!(
                "row {row}'s packed query offsets are {start}..{end}"
            )));
        }

        for token in start..end {
            if slots[token] == NO_SLOT {
                continue;
            }
            let position = u32::try_from(positions[token]).map_err(|_| {
                Error::Invariant(format!(
                    "token {token} writes cache position {}, which is no position",
                    positions[token]
                ))
            })?;

            let (page, offset) = (position / page_size, position % page_size);
            if page >= pages {
                return Err(Error::Invalid(format!(
                    "row {row} writes position {position}, which needs page {page} of a scratch \
                     set of {pages} pages a row; the shadow forward's page set is too small"
                )));
            }

            let flat = (row as u64 * u64::from(pages) + u64::from(page)) * u64::from(page_size)
                + u64::from(offset);
            mapping[token] = u32::try_from(flat)
                .ok()
                .filter(|slot| *slot != NO_SLOT)
                .ok_or_else(|| {
                    Error::Invalid(format!(
                        "a scratch slot of {flat} is outside the range a slot column holds"
                    ))
                })?;
        }
    }
    Ok(mapping)
}

/// The scratch slot column a shadow forward reads, over the **bucket's** `tokens` tokens.
///
/// A captured graph addresses the whole column, so it states a value for every token of the
/// bucket: the batch's `live` tokens by [`scratch_slot_mapping`], and [`NO_SLOT`] for the
/// tokens the batch does not hold, because a value left from an earlier batch would send the
/// shadow forward's writes into a page of another row. `query_offsets` is the batch's
/// `[rows + 1]` column, whose last entry is `live`; `positions` and `slots` hold at least the
/// live tokens.
///
/// # Errors
///
/// A batch of more live tokens than the bucket or than its columns hold, and what
/// [`scratch_slot_mapping`] refuses.
pub fn scratch_column(
    positions: &[i32],
    query_offsets: &[u32],
    slots: &[u32],
    live: usize,
    tokens: usize,
    pages: u32,
    page_size: u32,
) -> Result<Vec<u32>> {
    if live > positions.len() || live > slots.len() || tokens < live {
        return Err(Error::Invariant(format!(
            "a batch of {live} live token(s) in a bucket of {tokens} names {} position(s) and {} \
             slot(s)",
            positions.len(),
            slots.len()
        )));
    }

    let mut mapping = if live == 0 {
        Vec::new()
    } else {
        scratch_slot_mapping(
            &positions[..live],
            query_offsets,
            &slots[..live],
            pages,
            page_size,
        )?
    };
    mapping.resize(tokens, NO_SLOT);
    Ok(mapping)
}

#[cfg(test)]
mod tests {
    use ipc::PrefixCacheDtype;

    use super::*;

    /// A page of the set spans every column of the cache, a layer's cell being one column of
    /// one page: the figure is a BF16 page of 64 tokens over 48 layers of four heads of 128.
    #[test]
    fn a_page_of_the_set_spans_every_column() {
        let layout = CacheLayout::dense(64, 48, 4, 128, PrefixCacheDtype::BFloat16);
        let page = 48 * (2 * 65_536 + 2);
        let bytes = 1_237_319_680;
        let reserved = 7 * SCRATCH_GRANULARITY;

        assert_eq!(
            pages_within(&layout, bytes),
            (bytes - reserved) / (page + 4)
        );
        assert_eq!(pages_within(&layout, reserved), 0);
    }

    /// A set's planes have the cache's page shape over the set's pages, and its table reads at
    /// any width the set's pages cover.
    #[test]
    fn a_sets_tensors_are_its_pages_in_the_caches_shape() -> Result<()> {
        let layout = Arc::new(CacheLayout::dense(
            16,
            2,
            1,
            8,
            PrefixCacheDtype::Float8E4m3Fn,
        ));
        let set = ScratchPages::meta(Arc::clone(&layout), 12)?;
        assert_eq!(set.pages(), 12);

        let layer = set.layer(&layout.groups[0], 1)?;
        assert_eq!(layer.key.dims(), [12, 16, 1, 8]);
        assert_eq!(layer.key.dtype(), DType::F8E4M3);
        assert_eq!(layer.value_initialized.dims(), [12, 1, 1, 1]);
        assert!(layer.key_scale.is_some());
        assert!(set.layer(&layout.groups[0], 2).is_err());

        let table = set.table(3, 4)?;
        assert_eq!((table.dims(), table.dtype()), (&[3, 4][..], DType::U32));
        // Four rows of four pages are more than the set's twelve.
        assert!(set.table(4, 4).is_err());
        assert!(ScratchPages::meta(layout, 0).is_err());
        Ok(())
    }

    /// Every token of a shadow forward writes a page of its own row of the set.
    #[test]
    fn a_shadow_forward_writes_its_own_scratch_pages() -> Result<()> {
        let mapping =
            scratch_slot_mapping(&[5, 6, 7, 0, 1], &[0, 3, 5], &[10, 11, 12, 20, 21], 2, 4)?;

        // Row 0: positions 5..8 are page 1, offsets 1..3. Row 1: positions 0..2 of its page 0,
        // which is the set's page 2.
        assert_eq!(mapping, vec![5, 6, 7, 8, 9]);
        Ok(())
    }

    /// A token the served batch does not write writes no scratch page either, and a position
    /// past the row's pages is refused by name instead of wrapped into another row's.
    #[test]
    fn a_padding_token_writes_nothing_and_a_full_set_is_refused() -> Result<()> {
        let mapping = scratch_slot_mapping(&[0, 0], &[0, 2], &[NO_SLOT, 3], 1, 4)?;
        assert_eq!(mapping, vec![NO_SLOT, 0]);

        let error = scratch_slot_mapping(&[9], &[0, 1], &[7], 1, 4)
            .expect_err("position 9 needs page 2 of a one-page row");
        assert!(error.to_string().contains("page 2"), "{error}");

        // The three columns describe one batch.
        assert!(scratch_slot_mapping(&[0, 1], &[0, 3], &[7, 8], 1, 4).is_err());
        assert!(scratch_slot_mapping(&[-1], &[0, 1], &[7], 1, 4).is_err());
        Ok(())
    }

    /// A bucket wider than the batch names no page for the tokens the batch does not hold.
    #[test]
    fn a_partial_buckets_scratch_tail_names_no_page() -> Result<()> {
        let mapping = scratch_column(&[0, 1, 2], &[0, 3], &[10, 11, 12], 3, 8, 2, 4)?;
        assert_eq!(mapping.len(), 8);
        assert_eq!(&mapping[..3], &[0, 1, 2]);
        assert!(mapping[3..].iter().all(|slot| *slot == NO_SLOT));

        // A batch that holds nothing writes nothing, whatever the bucket's width.
        assert_eq!(
            scratch_column(&[], &[0], &[], 0, 4, 2, 4)?,
            vec![NO_SLOT; 4]
        );

        // Two live tokens do not fit a bucket of one, nor one a bucket of none.
        let error = scratch_column(&[0, 1], &[0, 2], &[7, 8], 2, 1, 1, 4)
            .expect_err("two live tokens in a bucket of one");
        assert!(error.to_string().contains("bucket of"), "{error}");
        assert!(scratch_column(&[0], &[0, 1], &[7], 1, 0, 1, 4).is_err());

        // A slot column shorter than the live count is a batch whose columns disagree.
        let error = scratch_column(&[0, 1], &[0, 2], &[7], 2, 2, 1, 4)
            .expect_err("two positions and one slot");
        assert!(error.to_string().contains("slot(s)"), "{error}");
        Ok(())
    }
}
