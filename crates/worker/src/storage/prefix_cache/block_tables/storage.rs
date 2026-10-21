//! Device block tables: native installs, clears and verified lengths.

use std::sync::Arc;

use uniserve_cuda::{CopyBatch, DeviceId, SourceOrder, Stream, memset_d32};

use super::{BlockTableUpdate, GroupShape};
use crate::ffi::{Backing, DType};
use crate::storage::memory::{Storage, StorageView};
use crate::stream::{Fill, Issue};
use crate::{Error, Result};

/// Byte alignment of each carved table inside the shared allocation.
const ALIGNMENT: usize = 256;

/// The numerical block tables of one rank, in one allocation:
///
/// | Table | Element | Shape |
/// | --- | --- | --- |
/// | `unit_tables` | `i32` | `[tables, rows, width]` |
/// | `start_pages` | `i32` | `[groups, rows]` |
/// | `alloced_lens` | `i32` | `[rows]` |
/// | `verified_lengths` | `i32` | `[rows]` |
/// | `table_shapes` | `i32` | `[tables, 3]`: group, block size, window or -1 |
///
/// `rows = request_pool_size + 1`; row 0 is the padding slot. Everything is
/// zero after allocation except the constant `table_shapes`. Writes are
/// stream-ordered and never issued on a capturing stream.
pub struct BlockTableStorage {
    memory: Arc<Storage>,
    device: Option<DeviceId>,
    rows: u32,
    tables: u32,
    groups: u32,
    width: u32,
    unit_tables: StorageView,
    start_pages: StorageView,
    alloced_lens: StorageView,
    verified_lengths: StorageView,
    table_shapes: StorageView,
    /// `width` zero words: the source of slot clears.
    zeros: StorageView,
}

/// Byte offsets of the carved tables and the total length.
fn layout(rows: usize, tables: usize, groups: usize, width: usize) -> ([usize; 6], usize) {
    let sizes = [
        tables * rows * width * 4,
        groups * rows * 4,
        rows * 4,
        rows * 4,
        tables * 3 * 4,
        width * 4,
    ];
    let mut offsets = [0; 6];
    let mut cursor = 0;
    for (offset, size) in offsets.iter_mut().zip(sizes) {
        *offset = cursor;
        cursor += size.next_multiple_of(ALIGNMENT);
    }
    (offsets, cursor)
}

impl BlockTableStorage {
    /// Device bytes of the tables of `groups` with `request_pool_size` slots
    /// and `width` pages per table.
    pub(crate) fn bytes(groups: &[GroupShape], request_pool_size: u32, width: u32) -> u64 {
        let tables: u32 = groups.iter().map(|group| group.units_per_page).sum();
        layout(
            request_pool_size as usize + 1,
            tables as usize,
            groups.len(),
            width as usize,
        )
        .1 as u64
    }

    /// Allocate zeroed tables on `device` (host memory when `None`), resetting
    /// them on `stream`, which is synchronized before returning.
    pub fn allocate(
        groups: &[GroupShape],
        request_pool_size: u32,
        width: u32,
        device: Option<DeviceId>,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        if groups.is_empty() || request_pool_size == 0 || width == 0 {
            return Err(Error::Invalid("block-table dimensions are invalid".into()));
        }
        if device.is_some() != stream.is_some() {
            return Err(Error::Invalid(
                "device block tables are reset on a stream of their device".into(),
            ));
        }
        let rows = request_pool_size + 1;
        let tables: u32 = groups.iter().map(|group| group.units_per_page).sum();
        let (offsets, bytes) = layout(rows as usize, tables as usize, groups.len(), width as usize);
        let memory = match device {
            Some(device) => Storage::device(device, bytes)?,
            None => Storage::host(bytes),
        };
        let backing = || Arc::clone(&memory) as Arc<dyn Backing>;
        let view = |index: usize, shape: &[i64]| {
            StorageView::contiguous(backing(), offsets[index], DType::I32, shape)
        };
        let (rows_i, tables_i, width_i) = (i64::from(rows), i64::from(tables), i64::from(width));

        let storage = Self {
            device,
            rows,
            tables,
            groups: groups.len() as u32,
            width,
            unit_tables: view(0, &[tables_i, rows_i, width_i])?,
            start_pages: view(1, &[groups.len() as i64, rows_i])?,
            alloced_lens: view(2, &[rows_i])?,
            verified_lengths: view(3, &[rows_i])?,
            table_shapes: view(4, &[tables_i, 3])?,
            zeros: view(5, &[width_i])?,
            memory,
        };

        // Constant shapes of every table, group-major.
        let shapes: Vec<i32> = groups
            .iter()
            .enumerate()
            .flat_map(|(group, shape)| {
                let row = [
                    group as i32,
                    shape.block_size as i32,
                    shape.window.map_or(-1, |window| window as i32),
                ];
                std::iter::repeat_n(row, shape.units_per_page as usize).flatten()
            })
            .collect();
        match stream {
            Some(stream) => {
                // SAFETY: the allocation spans `bytes`; the upload reads its
                // pageable source during the call; the stream drains below.
                unsafe {
                    memset_d32(storage.memory.address(), 0, bytes / 4, stream)?;
                    let mut batch = CopyBatch::new(SourceOrder::DuringCall);
                    batch.push(
                        storage.table_shapes.address(),
                        shapes.as_ptr() as u64,
                        shapes.len() * 4,
                    );
                    batch.submit(stream)?;
                }
                stream.synchronize()?;
            }
            None => {
                // SAFETY: fresh host memory owned by this value (already zero).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        shapes.as_ptr(),
                        storage.table_shapes.address() as *mut i32,
                        shapes.len(),
                    );
                }
            }
        }
        Ok(storage)
    }

    /// Per-slot verified cache lengths; the decode state reads and advances
    /// them as well.
    pub(crate) fn verified_lengths(&self) -> &StorageView {
        &self.verified_lengths
    }

    /// The tables keyed as numerical bindings receive them.
    pub fn views(&self) -> Vec<(String, StorageView)> {
        [
            ("unit_tables", &self.unit_tables),
            ("start_pages", &self.start_pages),
            ("table_shapes", &self.table_shapes),
            ("alloced_lens", &self.alloced_lens),
            ("verified_lengths", &self.verified_lengths),
        ]
        .into_iter()
        .map(|(name, view)| (name.to_owned(), view.clone()))
        .collect()
    }

    fn entry(&self, view: &StorageView, index: u64) -> u64 {
        view.address() + index * 4
    }

    fn row_address(&self, table: u32, slot: u32) -> u64 {
        self.entry(
            &self.unit_tables,
            (u64::from(table) * u64::from(self.rows) + u64::from(slot)) * u64::from(self.width),
        )
    }

    fn require_slot(&self, slot: u32) -> Result<()> {
        if slot == 0 || slot >= self.rows {
            return Err(Error::Invalid(
                "request slot is outside the block tables".into(),
            ));
        }
        Ok(())
    }

    /// Write a validated update (`BlockTables::prepare`) in one batched copy
    /// on `stream`: every changed row at full width (which also clears the
    /// tail of a longer previous row), then start pages and lengths. The
    /// host payload is consumed during the call, so installs never wait on
    /// the device. Commit the update on the host after this succeeds.
    pub fn install(&self, update: &BlockTableUpdate, stream: impl Issue) -> Result<()> {
        let width = self.width as usize;
        // One word per first page and extent, besides the rows.
        let words = update.starts.len() + update.extents.len();
        let mut payload: Vec<i32> = Vec::with_capacity(update.rows.len() * width + words);
        let mut entries = Vec::with_capacity(update.rows.len() + words);
        for row in &update.rows {
            let ((slot, _), table) = &update.tables[row.table];
            if table.pages() > width || row.target >= self.tables as usize {
                return Err(Error::Invalid("block-table row exceeds its table".into()));
            }
            self.require_slot(*slot)?;
            entries.push((
                self.row_address(row.target as u32, *slot),
                payload.len(),
                width,
            ));
            let end = payload.len() + width;
            payload.extend(table.row_units(row.position).map(|unit| unit as i32));
            payload.resize(end, 0);
        }
        for &(group, slot, value) in &update.starts {
            if group >= self.groups {
                return Err(Error::Invalid("block-table group is out of range".into()));
            }
            self.require_slot(slot)?;
            let index = u64::from(group) * u64::from(self.rows) + u64::from(slot);
            entries.push((self.entry(&self.start_pages, index), payload.len(), 1));
            payload.push(value as i32);
        }
        for &(slot, value) in &update.extents {
            self.require_slot(slot)?;
            entries.push((
                self.entry(&self.alloced_lens, u64::from(slot)),
                payload.len(),
                1,
            ));
            payload.push(value as i32);
        }
        self.write(&entries, &payload, stream)
    }

    /// Copy `payload` words to their destinations: `(address, first word,
    /// words)` entries with disjoint destinations.
    fn write(
        &self,
        entries: &[(u64, usize, usize)],
        payload: &[i32],
        stream: impl Issue,
    ) -> Result<()> {
        if self.device.is_some() != stream.stream().is_some() {
            return Err(Error::Invalid(
                "block-table writes use a stream exactly for device tables".into(),
            ));
        }
        let mut batch = CopyBatch::new(SourceOrder::DuringCall);
        for &(address, first, words) in entries {
            batch.push(address, payload[first..].as_ptr() as u64, words * 4);
        }
        // SAFETY: destinations lie inside this allocation and never overlap
        // (one entry per changed row, start and slot length); the payload
        // is read during the call. Host tables are written in the order of
        // their readers.
        unsafe { stream.copy(&batch) }
    }

    /// Zero released slots' rows, start pages and lengths on `stream`, which
    /// must be ordered after every batch that reads those rows.
    pub fn clear(&self, slots: &[u32], stream: impl Issue) -> Result<()> {
        let mut entries: Vec<(u64, u64, usize)> = Vec::new();
        for &slot in slots {
            self.require_slot(slot)?;
            for table in 0..self.tables {
                entries.push((
                    self.row_address(table, slot),
                    self.zeros.address(),
                    self.width as usize * 4,
                ));
            }
            for group in 0..self.groups {
                let index = u64::from(group) * u64::from(self.rows) + u64::from(slot);
                entries.push((
                    self.entry(&self.start_pages, index),
                    self.zeros.address(),
                    4,
                ));
            }
            entries.push((
                self.entry(&self.alloced_lens, u64::from(slot)),
                self.zeros.address(),
                4,
            ));
            entries.push((
                self.entry(&self.verified_lengths, u64::from(slot)),
                self.zeros.address(),
                4,
            ));
        }
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for &(destination, source, bytes) in &entries {
            batch.push(destination, source, bytes);
        }
        // SAFETY: destinations are distinct entries of this allocation
        // (callers deduplicate slots) and the zero row is never written.
        unsafe { stream.copy(&batch) }
    }

    /// Set a slot's verified cache length after an import installed `extent`
    /// tokens. The caller checked `extent` against the slot's allocation.
    pub fn set_verified(&self, slot: u32, extent: u32, stream: impl Issue) -> Result<()> {
        self.require_slot(slot)?;
        let address = self.entry(&self.verified_lengths, u64::from(slot));
        // SAFETY: one aligned word of this allocation.
        unsafe {
            stream.fill(Fill {
                address,
                value: extent,
                width: 4,
                count: 1,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::BlockTables;
    use uniserve_ipc::BlockTable;
    use uniserve_ipc::UnitId;

    fn read(view: &StorageView) -> Vec<i32> {
        let count = view.numel() as usize;
        // SAFETY: host tables of the test, not written concurrently.
        unsafe { std::slice::from_raw_parts(view.address() as *const i32, count).to_vec() }
    }

    #[test]
    fn host_installs_write_full_rows_and_clears_reset_slots() -> Result<()> {
        let groups = vec![
            GroupShape::new(4, 1, None)?,
            GroupShape::new(8, 2, Some(8))?,
        ];
        let storage = BlockTableStorage::allocate(&groups, 2, 3, None, None)?;
        assert_eq!(
            read(&storage.table_shapes),
            vec![0, 4, -1, 1, 8, 8, 1, 8, 8]
        );

        let mut tables = BlockTables::new(groups, 2, 3, 16)?;
        let update = tables.prepare(&[
            BlockTable {
                request_pool_idx: 2,
                group_id: 0,
                start_page: 0,
                unit_ids: vec![UnitId(5), UnitId(6), UnitId(7)],
                allocated_tokens: 12,
            },
            BlockTable {
                request_pool_idx: 2,
                group_id: 1,
                start_page: 1,
                unit_ids: vec![UnitId(8), UnitId(9)],
                allocated_tokens: 16,
            },
        ])?;
        storage.install(&update, None)?;
        tables.commit(update);

        let units = read(&storage.unit_tables);
        // Table 0 (group 0), slot 2: the full row.
        assert_eq!(&units[2 * 3..3 * 3], &[5, 6, 7]);
        // Tables 1 and 2 hold group 1's two unit positions, zero-padded.
        assert_eq!(&units[(3 + 2) * 3..(3 + 3) * 3], &[8, 0, 0]);
        assert_eq!(&units[(6 + 2) * 3..(6 + 3) * 3], &[9, 0, 0]);
        assert_eq!(read(&storage.start_pages), vec![0, 0, 0, 0, 0, 1]);
        assert_eq!(read(&storage.alloced_lens), vec![0, 0, 12]);

        storage.set_verified(2, 7, None)?;
        assert_eq!(read(storage.verified_lengths()), vec![0, 0, 7]);
        storage.clear(&[2], None)?;
        assert!(read(&storage.unit_tables).iter().all(|&unit| unit == 0));
        assert!(
            read(&storage.alloced_lens)
                .iter()
                .all(|&length| length == 0)
        );
        assert!(
            read(storage.verified_lengths())
                .iter()
                .all(|&length| length == 0)
        );
        assert!(storage.clear(&[3], None).is_err());
        Ok(())
    }
}
