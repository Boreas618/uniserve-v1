//! The rank's pooled page tables: the device copy of the attention groups' entries of the rank's
//! device block table, which a forward gathers its rows from.
//!
//! The buffer pool's completions feed the rank's device block table (`memory::tables`), and its
//! changes are written here, each once, before the batch that reads them. For every
//! **attention** group the cache keeps one pooled table `[slots + 1, width]` of `u32`:
//!
//! - row `slot - 1` is request slot `slot`'s (slots count from one);
//! - entry `p` of a row is the unit of the block at position `p` of the slot's table of the
//!   group's class, which is the row's logical page `p`, so a forward addresses a token at
//!   cache position `q` through entry `q / block_size`; a position that holds no entry names
//!   unit 0, the padding unit;
//! - the last row, row `slots`, names only the padding unit: a bucket's padding rows gather it.
//!
//! The pooled tables are on the host always and mirrored on the device of a cache that holds
//! one. The state group's and the draft group's blocks are addressed through the device block
//! table itself, by the copies that name a checkpoint's position and by the tables a
//! speculative step builds, so neither has a pooled table.

use std::collections::BTreeSet;
use std::sync::Arc;

use platform::{CopyBatch, DeviceId, SourceOrder, Stream, memset_d32};
use uniserve_tensor::{DType, Tensor};

use super::layout::CacheLayout;
use crate::memory::allocation::Storage;
use crate::memory::lend;
use crate::memory::tables::Change;
use crate::{Error, Result};

/// The pooled tables of the attention groups.
pub(crate) struct PageTables {
    layout: Arc<CacheLayout>,
    /// Request slots: slots `1..=slots` own rows.
    slots: u32,
    /// Pages per row: the pages a request of the model's length holds.
    width: u32,
    /// Units of the pool, the padding unit included: the bound of every unit a table names.
    units: u32,
    /// Each attention group's pooled table on the host, `[slots + 1, width]` row-major, by the
    /// group's index among the attention groups.
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
                // row writes this value issues write it, in stream order.
                unsafe { storage.external(offset, shape.as_slice(), DType::U32) }
            }
            // SAFETY: a meta tensor has no data and its address is never read.
            None => unsafe { lend(0, shape.as_slice(), DType::U32, None) },
        }
    }

    /// The pooled table of attention group `group` on the host, `[slots + 1, width]` row-major:
    /// what the device table holds once the changes applied so far completed.
    pub(crate) fn host_table(&self, group: usize) -> Result<&[u32]> {
        self.rows
            .get(group)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::Invalid(format!("the cache has no attention group {group}")))
    }

    /// Writes `changes`, the device block table's changes since the last batch, into the
    /// pooled tables: each change of an attention group's class sets the entry at its position
    /// of its slot's row to its run's unit, or to the padding unit where the position holds no
    /// entry; a change of any other class has no pooled table and is skipped. The class of
    /// cache group `g` is class `g`. Every change is checked first, and nothing changes when one
    /// is refused. The rows that changed are written at full width, on the host and, for
    /// tables on a device, in one batched copy on `stream`, whose host source is read during
    /// the call.
    ///
    /// # Errors
    ///
    /// A change of a slot outside the tables' slots, of a position past a row's pages, or of a
    /// run that is not one unit of the pool; tables on a device written without a stream, or
    /// the reverse.
    pub(crate) fn apply(&mut self, changes: &[Change], stream: Option<&Stream>) -> Result<()> {
        if self.device.is_some() != stream.is_some() {
            return Err(Error::Invalid(
                "page tables are written on a stream exactly when they are on a device".into(),
            ));
        }

        // Each attention change as `(attention group, row, entry, unit)`.
        let mut entries = Vec::new();
        for change in changes {
            let Some(group) = self.attention_index(change.class) else {
                continue;
            };
            let refusal = |reason: String| {
                Error::Invalid(format!(
                    "the change of position {} of class {} of request slot {} is refused: \
                     {reason}",
                    change.position, change.class, change.slot
                ))
            };

            let row = self
                .row(change.slot)
                .map_err(|error| refusal(error.to_string()))?;
            if change.position >= self.width {
                return Err(refusal(format!("a row holds {} pages", self.width)));
            }
            let unit = match change.run {
                None => 0,
                Some((_, unit, 1)) if unit != 0 && unit < self.units => unit,
                Some((_, first, count)) => {
                    return Err(refusal(format!(
                        "its run of {count} units from unit {first} is not one unit of the \
                         pool's units 1..{}",
                        self.units
                    )));
                }
            };
            entries.push((group, row as usize, change.position as usize, unit));
        }

        let width = self.width as usize;
        let mut changed = BTreeSet::new();
        for &(group, row, entry, unit) in &entries {
            self.rows[group][row * width + entry] = unit;
            changed.insert((group, row));
        }

        if let (Some(storage), Some(stream)) = (&self.device, stream) {
            let table_bytes = (self.slots as usize + 1) * width * 4;
            let mut batch = CopyBatch::new(SourceOrder::DuringCall);
            for &(group, row) in &changed {
                batch.push(
                    storage.address() + (group * table_bytes + row * width * 4) as u64,
                    self.rows[group][row * width..].as_ptr() as u64,
                    width * 4,
                );
            }

            // SAFETY: the destinations are distinct rows of the storage the tables own (one
            // per slot and group), and the host rows are read during the call.
            unsafe { batch.submit(stream) }?;
        }
        Ok(())
    }

    /// The index among the attention groups of the cache group whose class is `class`, when it
    /// is an attention group: they come first in the layout.
    fn attention_index(&self, class: u8) -> Option<usize> {
        let group = self.layout.groups.get(class as usize)?;
        (group.attention() && (class as usize) < self.rows.len()).then_some(class as usize)
    }
}

#[cfg(test)]
mod tests {
    use ipc::PrefixCacheDtype;

    use super::*;

    /// The region of the units the changes name.
    const UNITS: u8 = 0;

    /// Tables of two request slots and rows of four pages over sixteen units: a cache of two
    /// attention layers of 4-token pages (class 0) and a recurrent state of two layers whose
    /// pages span two units (class 1).
    fn tables() -> PageTables {
        let state = nn::family::StateShape {
            layers: 2,
            fields: vec![nn::family::StateField {
                name: "state",
                dtype: DType::U8,
                dims: vec![40],
                stepwise: false,
            }],
        };
        let layout = CacheLayout::hybrid(4, 2, 1, 4, PrefixCacheDtype::BFloat16, &state)
            .expect("a hybrid layout");
        assert!(layout.groups[1].state() && layout.groups[1].units_per_page == 2);

        PageTables::meta(Arc::new(layout), 2, 4, 16).expect("page tables")
    }

    /// The change of `position` of `class` of `slot` to the block of `units` units from unit
    /// `first`, or to no entry for `None`.
    fn change(slot: u32, class: u8, position: u32, block: Option<(u32, u32)>) -> Change {
        Change {
            slot,
            class,
            position,
            run: block.map(|(first, units)| (UNITS, first, units)),
            reset: false,
        }
    }

    /// A slot's row is the unit of each block its table holds at that position, the padding
    /// unit everywhere else, and the last row padding alone.
    #[test]
    fn a_slots_row_is_the_units_of_its_entries() -> Result<()> {
        let mut pooled = tables();
        assert_eq!(pooled.table(0)?.dims(), [3, 4]);
        assert_eq!(pooled.table(0)?.dtype(), DType::U32);
        assert_eq!(
            (pooled.row(1)?, pooled.row(2)?, pooled.padding_row()),
            (0, 1, 2)
        );

        let held: Vec<Change> = (0..3)
            .map(|position| change(2, 0, position, Some((5 + position, 1))))
            .collect();
        pooled.apply(&held, None)?;
        assert_eq!(pooled.host_table(0)?, [0, 0, 0, 0, 5, 6, 7, 0, 0, 0, 0, 0]);

        // Entries that left behind a window name the padding unit again, at their positions,
        // and another slot's entries are its row's.
        let changes = [
            change(2, 0, 0, None),
            change(2, 0, 1, None),
            change(1, 0, 2, Some((3, 1))),
            change(1, 0, 3, Some((4, 1))),
        ];
        pooled.apply(&changes, None)?;
        assert_eq!(pooled.host_table(0)?, [0, 0, 3, 4, 0, 0, 7, 0, 0, 0, 0, 0]);
        Ok(())
    }

    /// A state block spans two units and has no pooled table: its entries change no row.
    #[test]
    fn a_state_entry_has_no_pooled_row() -> Result<()> {
        let mut pooled = tables();

        pooled.apply(&[change(1, 1, 1, Some((8, 2)))], None)?;
        assert!(pooled.host_table(0)?.iter().all(|unit| *unit == 0));
        assert!(pooled.table(1).is_err());
        Ok(())
    }

    /// A refused change names its slot, its class and its position, and writes nothing, not
    /// even the valid changes beside it.
    #[test]
    fn a_refused_change_writes_nothing() -> Result<()> {
        let mut pooled = tables();
        let valid = change(1, 0, 0, Some((1, 1)));
        let mut refused = |other: Change| {
            pooled
                .apply(&[valid, other], None)
                .expect_err("the change is refused")
                .to_string()
        };

        // Slot zero owns no row, and neither does a slot past the last.
        let slot = refused(change(0, 0, 0, Some((3, 1))));
        assert!(slot.contains("request slot 0"), "{slot}");
        assert!(refused(change(3, 0, 0, Some((3, 1)))).contains("1..=2"));
        // A position past a row's pages.
        assert!(refused(change(2, 0, 4, Some((3, 1)))).contains("4 pages"));
        // A unit outside the pool, the padding unit, and a block of two units.
        assert!(refused(change(2, 0, 0, Some((16, 1)))).contains("unit 16"));
        assert!(refused(change(2, 0, 0, Some((0, 1)))).contains("unit 0"));
        assert!(refused(change(2, 0, 0, Some((3, 2)))).contains("2 units"));

        assert!(pooled.host_table(0)?.iter().all(|unit| *unit == 0));
        Ok(())
    }
}
