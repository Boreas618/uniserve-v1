//! The rank's page tables: the scheduler's page assignments and the pooled tables a forward
//! gathers its rows from.
//!
//! The scheduler owns allocation and states each request slot's pages per cache group
//! (`ipc::BlockTable`); this module mirrors them. For every group it keeps what each slot holds,
//! and for every **attention** group it keeps one pooled table `[slots + 1, width]` of `u32`:
//!
//! - row `slot - 1` is request slot `slot`'s (slots count from one);
//! - entry `p` of a row is the unit of the row's logical page `p`, so a forward addresses a
//!   token at position `q` through entry `q / block_size`; a page the slot does not hold (one
//!   before the table's first page, one past its last) names unit 0, the padding unit;
//! - the last row, row `slots`, names only the padding unit: a bucket's padding rows gather it.
//!
//! The pooled tables are on the host always and mirrored on the device of a cache that holds
//! one. The state group's pages are addressed by the copies that name their units and the
//! draft group's through the tables a speculative step builds from
//! [`PageTables::installed`], so neither has a pooled table.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ipc::BlockTable;
use platform::{CopyBatch, DeviceId, SourceOrder, Stream, memset_d32};
use uniserve_tensor::{DType, Tensor};

use super::layout::CacheLayout;
use crate::memory::allocation::Storage;
use crate::memory::lend;
use crate::{Error, Result};

/// One slot's pages in one cache group, as the scheduler last stated them.
struct Installed {
    /// The first logical page the units cover.
    start_page: u32,
    /// The units of pages `start_page..`, page-major.
    units: Vec<u32>,
}

/// The host mirror of the scheduler's page tables and the pooled tables of the attention
/// groups.
pub(crate) struct PageTables {
    layout: Arc<CacheLayout>,
    /// Request slots: slots `1..=slots` own rows.
    slots: u32,
    /// Pages per row: the pages a request of the model's length holds.
    width: u32,
    /// Units of the pool, the padding unit included: the bound of every unit a table names.
    units: u32,
    installed: HashMap<(u32, u32), Installed>,
    /// Each attention group's pooled table on the host, `[slots + 1, width]` row-major.
    rows: Vec<Vec<u32>>,
    /// The pooled tables on the device, group after group in one storage, each
    /// `[slots + 1, width]`; `None` for tables that hold no device.
    device: Option<Arc<Storage>>,
}

impl PageTables {
    /// The device bytes of the pooled tables of `layout` for `slots` request slots and rows of
    /// `width` pages.
    pub(crate) fn device_bytes(layout: &CacheLayout, slots: u32, width: u32) -> u64 {
        layout.attention_groups() as u64 * (u64::from(slots) + 1) * u64::from(width) * 4
    }

    /// Tables for `slots` request slots and rows of `width` pages over a pool of `units` units,
    /// with no device: the pooled tables are on the host alone.
    pub(crate) fn meta(
        layout: Arc<CacheLayout>,
        slots: u32,
        width: u32,
        units: u32,
    ) -> Result<Self> {
        if slots == 0 || width == 0 || units == 0 {
            return Err(Error::Invalid(
                "page tables need request slots, pages per row and units".into(),
            ));
        }

        let entries = (slots as usize + 1) * width as usize;
        let rows = vec![vec![0; entries]; layout.attention_groups()];

        Ok(Self {
            layout,
            slots,
            width,
            units,
            installed: HashMap::new(),
            rows,
            device: None,
        })
    }

    /// Tables as [`Self::meta`] makes them, mirrored on `device`: the pooled tables are
    /// allocated there and zeroed on `stream`, which is synchronized before returning.
    pub(crate) fn open(
        layout: Arc<CacheLayout>,
        slots: u32,
        width: u32,
        units: u32,
        device: DeviceId,
        stream: &Stream,
    ) -> Result<Self> {
        let mut tables = Self::meta(layout, slots, width, units)?;

        let bytes = Self::device_bytes(&tables.layout, slots, width) as usize;
        let storage = Storage::linear(device, bytes)?;

        // SAFETY: the fill spans exactly the new allocation, which the tables keep; the stream
        // is drained before anything reads it.
        unsafe { memset_d32(storage.address(), 0, bytes / 4, stream) }?;
        stream.synchronize()?;

        tables.device = Some(storage);
        Ok(tables)
    }

    /// Pages per row.
    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    /// The pooled table's row that request slot `slot` owns.
    ///
    /// # Errors
    ///
    /// For a slot outside `1..=slots`.
    pub(crate) fn row(&self, slot: u32) -> Result<u32> {
        if slot == 0 || slot > self.slots {
            return Err(Error::Invalid(format!(
                "request slot {slot} is outside the page tables' slots 1..={}",
                self.slots
            )));
        }
        Ok(slot - 1)
    }

    /// The pooled table's last row, which names only the padding unit.
    pub(crate) fn padding_row(&self) -> u32 {
        self.slots
    }

    /// The pooled table of attention group `group` as a `[slots + 1, width]` U32 tensor: on the
    /// device, or a meta tensor for tables that hold none.
    pub(crate) fn table(&self, group: usize) -> Result<Tensor> {
        if group >= self.rows.len() {
            return Err(Error::Invalid(format!(
                "the cache has no attention group {group}"
            )));
        }
        let shape = [self.slots as usize + 1, self.width as usize];

        match &self.device {
            Some(storage) => {
                let offset = group * shape[0] * shape[1] * 4;

                // SAFETY: the tables own the storage for as long as they live, and its holder
                // keeps the tables for as long as a tensor over them is in use; only the
                // installs and releases this value issues write it, in stream order.
                unsafe { storage.external(offset, shape.as_slice(), DType::U32) }
            }
            // SAFETY: a meta tensor has no data and its address is never read.
            None => unsafe { lend(0, shape.as_slice(), DType::U32, None) },
        }
    }

    /// The pooled table of attention group `group` on the host, `[slots + 1, width]` row-major:
    /// what the device table holds once the installs issued so far completed.
    pub(crate) fn host_table(&self, group: usize) -> Result<&[u32]> {
        self.rows
            .get(group)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::Invalid(format!("the cache has no attention group {group}")))
    }

    /// The pages `slot`'s installed table of `group` holds: its first logical page and the
    /// units of the pages from there on, page-major. `None` when the slot holds no table of the
    /// group.
    pub(crate) fn installed(&self, slot: u32, group: u32) -> Option<(u32, &[u32])> {
        self.installed
            .get(&(slot, group))
            .map(|table| (table.start_page, table.units.as_slice()))
    }

    /// Checks one of the scheduler's tables against the rank's own cache: a request slot, a
    /// group the layout holds, whole pages within a row, units inside the pool, and allocated
    /// tokens within its pages. The scheduler's allocator owns which units a table names; the
    /// pool bound stays here because kernels address the planes by these ids.
    fn check(&self, table: &BlockTable) -> Result<()> {
        let refusal = |reason: String| {
            Error::Invalid(format!(
                "the block table of request slot {} in cache group {} is refused: {reason}",
                table.request_pool_idx, table.group_id
            ))
        };

        self.row(table.request_pool_idx)
            .map_err(|error| refusal(error.to_string()))?;
        let group = self
            .layout
            .groups
            .get(table.group_id as usize)
            .ok_or_else(|| refusal("the cache holds no such group".into()))?;

        let per_page = group.units_per_page as usize;
        if !table.unit_ids.len().is_multiple_of(per_page) {
            return Err(refusal(format!(
                "{} units are not whole pages of {per_page}",
                table.unit_ids.len()
            )));
        }
        if let Some(unit) = table.unit_ids.iter().find(|unit| unit.0 >= self.units) {
            return Err(refusal(format!(
                "unit {} is outside the pool of {} units",
                unit.0, self.units
            )));
        }

        // An attention row is indexed by logical page, so its pages lie inside the row from
        // page zero; any other group's table is bounded by the pages it holds.
        let pages = (table.unit_ids.len() / per_page) as u64;
        let end_page = u64::from(table.start_page) + pages;
        let extent = if group.attention() { end_page } else { pages };
        if extent > u64::from(self.width) {
            return Err(refusal(format!(
                "its pages reach page {extent} of rows of {} pages",
                self.width
            )));
        }
        if u64::from(table.allocated_tokens) > end_page * u64::from(group.block_size) {
            return Err(refusal(format!(
                "{} allocated tokens exceed its {end_page} pages of {} tokens",
                table.allocated_tokens, group.block_size
            )));
        }
        if group.attention() && group.units_per_page != 1 {
            return Err(refusal(
                "an attention page of several units has no single unit a row names".into(),
            ));
        }
        Ok(())
    }

    /// Installs the scheduler's tables of a batch: checks them all, then writes the changed
    /// rows of the attention groups' pooled tables, on the host and, for tables on a device, in
    /// one batched copy on `stream`, whose host source is read during the call. A row is
    /// written at full width, which clears the tail a longer previous table left.
    ///
    /// Nothing changes when a table is refused or the copy fails.
    ///
    /// # Errors
    ///
    /// A table the rank's cache refuses, by slot, group and reason; two tables of one slot and
    /// group; tables on a device installed without a stream, or the reverse.
    pub(crate) fn install(&mut self, tables: &[BlockTable], stream: Option<&Stream>) -> Result<()> {
        if self.device.is_some() != stream.is_some() {
            return Err(Error::Invalid(
                "page tables are installed on a stream exactly when they are on a device".into(),
            ));
        }

        let mut seen = HashSet::new();
        for table in tables {
            self.check(table)?;
            if !seen.insert((table.request_pool_idx, table.group_id)) {
                return Err(Error::Invalid(format!(
                    "a batch states two block tables of request slot {} in cache group {}",
                    table.request_pool_idx, table.group_id
                )));
            }
        }

        // The rows that change, as `(group, row, first entry in `payload`)`.
        let width = self.width as usize;
        let mut payload: Vec<u32> = Vec::new();
        let mut changed = Vec::new();
        for table in tables {
            let group = table.group_id as usize;
            if !self.layout.groups[group].attention() {
                continue;
            }
            let row = self.row(table.request_pool_idx)? as usize;

            let first = payload.len();
            payload.resize(first + width, 0);
            for (page, unit) in table.unit_ids.iter().enumerate() {
                payload[first + table.start_page as usize + page] = unit.0;
            }

            if payload[first..] == self.rows[group][row * width..(row + 1) * width] {
                payload.truncate(first);
            } else {
                changed.push((group, row, first));
            }
        }

        if let (Some(storage), Some(stream)) = (&self.device, stream) {
            let table_bytes = (self.slots as usize + 1) * width * 4;
            let mut batch = CopyBatch::new(SourceOrder::DuringCall);
            for &(group, row, first) in &changed {
                batch.push(
                    storage.address() + (group * table_bytes + row * width * 4) as u64,
                    payload[first..].as_ptr() as u64,
                    width * 4,
                );
            }

            // SAFETY: the destinations are distinct rows of the storage the tables own (one
            // per slot and group, checked above), and the payload is read during the call.
            unsafe { batch.submit(stream) }?;
        }

        for (group, row, first) in changed {
            self.rows[group][row * width..(row + 1) * width]
                .copy_from_slice(&payload[first..first + width]);
        }
        for table in tables {
            self.installed.insert(
                (table.request_pool_idx, table.group_id),
                Installed {
                    start_page: table.start_page,
                    units: table.unit_ids.iter().map(|unit| unit.0).collect(),
                },
            );
        }
        Ok(())
    }

    /// Forgets the tables of `slots` and names the padding unit in their rows again, on
    /// `stream` for tables on a device, ordered behind every batch that reads those rows.
    ///
    /// # Errors
    ///
    /// A slot outside the tables; tables on a device released without a stream, or the reverse.
    pub(crate) fn release(&mut self, slots: &[u32], stream: Option<&Stream>) -> Result<()> {
        if self.device.is_some() != stream.is_some() {
            return Err(Error::Invalid(
                "page tables are released on a stream exactly when they are on a device".into(),
            ));
        }
        let rows = slots
            .iter()
            .map(|slot| self.row(*slot).map(|row| row as usize))
            .collect::<Result<Vec<_>>>()?;

        let width = self.width as usize;
        if let (Some(storage), Some(stream)) = (&self.device, stream) {
            let table_bytes = (self.slots as usize + 1) * width * 4;
            for group in 0..self.rows.len() {
                for row in &rows {
                    let address =
                        storage.address() + (group * table_bytes + row * width * 4) as u64;

                    // SAFETY: one row of the storage the tables own.
                    unsafe { memset_d32(address, 0, width, stream) }?;
                }
            }
        }

        for (slot, row) in slots.iter().zip(rows) {
            for table in &mut self.rows {
                table[row * width..(row + 1) * width].fill(0);
            }
            for group in 0..self.layout.groups.len() as u32 {
                self.installed.remove(&(*slot, group));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ipc::{CacheGroupKind, PrefixCacheDtype, UnitId};

    use super::*;

    /// Tables of two request slots and rows of four pages over sixteen units: a cache of two
    /// attention layers of 4-token pages and a recurrent state of two layers.
    fn tables() -> PageTables {
        let state = nn::family::StateShape {
            layers: 2,
            fields: vec![nn::family::StateField {
                name: "state",
                dtype: DType::U8,
                dims: vec![40],
            }],
        };
        let layout = CacheLayout::hybrid(4, 2, 1, 4, PrefixCacheDtype::BFloat16, &state)
            .expect("a hybrid layout");
        assert!(layout.groups[1].state() && layout.groups[1].units_per_page == 2);

        PageTables::meta(Arc::new(layout), 2, 4, 16).expect("page tables")
    }

    fn table(slot: u32, group: u32, start_page: u32, units: &[u32], tokens: u32) -> BlockTable {
        BlockTable {
            request_pool_idx: slot,
            group_id: group,
            start_page,
            unit_ids: units.iter().copied().map(UnitId).collect(),
            allocated_tokens: tokens,
        }
    }

    /// An installed table is the row its slot's forward gathers: the units at their logical
    /// pages, the padding unit everywhere else, and the last row padding alone.
    #[test]
    fn an_installed_table_is_its_slots_row() -> Result<()> {
        let mut tables = tables();
        assert_eq!(tables.table(0)?.dims(), [3, 4]);
        assert_eq!(tables.table(0)?.dtype(), DType::U32);
        assert_eq!(
            (tables.row(1)?, tables.row(2)?, tables.padding_row()),
            (0, 1, 2)
        );

        tables.install(&[table(2, 0, 0, &[5, 6, 7], 12)], None)?;
        assert_eq!(tables.host_table(0)?, [0, 0, 0, 0, 5, 6, 7, 0, 0, 0, 0, 0]);
        assert_eq!(tables.installed(2, 0), Some((0, &[5, 6, 7][..])));
        assert_eq!(tables.installed(1, 0), None);

        // A shorter table of the same slot clears the tail the longer one left.
        tables.install(&[table(2, 0, 0, &[9], 4)], None)?;
        assert_eq!(tables.host_table(0)?[4..8], [9, 0, 0, 0]);

        // A table whose first pages were retired names its units at their logical pages.
        tables.install(&[table(1, 0, 2, &[3, 4], 16)], None)?;
        assert_eq!(tables.host_table(0)?[..4], [0, 0, 3, 4]);

        // The padding row is never written.
        assert_eq!(tables.host_table(0)?[8..], [0, 0, 0, 0]);
        Ok(())
    }

    /// The state group's table is kept for the copies that name its pages, and has no pooled
    /// table: a page that holds no checkpoint names the padding unit.
    #[test]
    fn a_state_table_is_mirrored_without_a_pooled_table() -> Result<()> {
        let mut tables = tables();

        tables.install(&[table(1, 1, 1, &[0, 0, 8, 9], 12)], None)?;
        assert_eq!(tables.installed(1, 1), Some((1, &[0, 0, 8, 9][..])));
        assert!(tables.host_table(0)?.iter().all(|unit| *unit == 0));
        assert!(tables.table(1).is_err());
        Ok(())
    }

    /// A refused table names its slot, its group and the reason, and installs nothing, not even
    /// the valid tables stated beside it.
    #[test]
    fn a_refused_table_installs_nothing() -> Result<()> {
        let mut tables = tables();
        let valid = table(1, 0, 0, &[1, 2], 8);
        let refused = |tables: &mut PageTables, table: BlockTable| {
            tables
                .install(&[valid.clone(), table], None)
                .expect_err("the table is refused")
                .to_string()
        };

        // Slot zero owns no row, and neither does a slot past the last.
        let slot = refused(&mut tables, table(0, 0, 0, &[3], 4));
        assert!(slot.contains("request slot 0"), "{slot}");
        assert!(refused(&mut tables, table(3, 0, 0, &[3], 4)).contains("1..=2"));
        // A group the cache does not hold.
        assert!(refused(&mut tables, table(2, 5, 0, &[3], 4)).contains("no such group"));
        // A unit outside the pool.
        assert!(refused(&mut tables, table(2, 0, 0, &[16], 4)).contains("unit 16"));
        // More pages than a row holds, counted from page zero.
        assert!(refused(&mut tables, table(2, 0, 3, &[3, 4], 20)).contains("page 5"));
        // Tokens its pages do not hold.
        assert!(refused(&mut tables, table(2, 0, 0, &[3], 5)).contains("5 allocated tokens"));
        // A state page of two units stated as one.
        assert!(refused(&mut tables, table(2, 1, 0, &[3], 4)).contains("whole pages"));
        // Two tables of one slot and group.
        assert!(refused(&mut tables, valid.clone()).contains("two block tables"));

        assert_eq!(tables.installed(1, 0), None);
        assert!(tables.host_table(0)?.iter().all(|unit| *unit == 0));
        Ok(())
    }

    /// A released slot's rows name the padding unit again and its tables are forgotten.
    #[test]
    fn a_released_slot_holds_nothing() -> Result<()> {
        let mut tables = tables();
        tables.install(
            &[
                table(1, 0, 0, &[1, 2], 8),
                table(1, 1, 0, &[4, 5], 4),
                table(2, 0, 0, &[3], 4),
            ],
            None,
        )?;

        tables.release(&[1], None)?;
        assert_eq!(tables.installed(1, 0), None);
        assert_eq!(tables.installed(1, 1), None);
        assert_eq!(tables.host_table(0)?, [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]);
        assert!(tables.release(&[3], None).is_err());
        Ok(())
    }

    /// An attention group whose page spans several units has no row to state.
    #[test]
    fn an_attention_page_of_several_units_is_refused() {
        let mut layout = CacheLayout::dense(4, 2, 1, 4, PrefixCacheDtype::BFloat16);
        layout.groups[0].units_per_page = 2;
        layout.groups[0].kind = CacheGroupKind::SlidingWindow { window: 8 };
        let mut tables = PageTables::meta(Arc::new(layout), 2, 4, 16).expect("page tables");

        let refused = tables
            .install(&[table(1, 0, 0, &[1, 2], 4)], None)
            .expect_err("a page of two units");
        assert!(refused.to_string().contains("several units"), "{refused}");
    }
}
