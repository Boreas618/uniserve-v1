//! The KV slot each query token of a call writes (design §9: the mechanism
//! writes the batch in the layout the kernels read).
//!
//! A request's KV lives in physical cache units, one per page, and a page holds
//! `page_size` tokens. The KV-write kernels address one row of the cache planes
//! per token (`slot_mapping + table * write_stride + token`), so the mechanism
//! must turn a request's block table and the positions of the tokens it
//! evaluates into those slots. Nothing else derives it: the executor submits
//! the column, it does not build it.
//!
//! The mapping is a pure function of the table and the positions, which is why
//! it lives here rather than in the call-building code: the assembly of a batch
//! and the test of the mapping can both use it.

use std::ops::Range;

/// Tokens one physical cache unit holds, from the table it covers.
///
/// A table states the absolute token extent it covers and one unit per page, so
/// their ratio is the page's token capacity. `None` when the table covers no
/// tokens or names no units, which is a table no call can write through.
pub(crate) fn page_size(table: &ipc::BlockTable) -> Option<u32> {
    let units = u32::try_from(table.unit_ids.len()).ok()?;
    let tokens = table.allocated_tokens;
    if units == 0 || tokens == 0 {
        return None;
    }
    let size = tokens / units;
    (size * units == tokens && size > 0).then_some(size)
}

/// The slot each position of `positions` writes in `table`, in order.
///
/// A position is absolute in its request's sequence. The page holding it is
/// `position / page_size`; the table's `start_page` is the first page its units
/// cover, so the unit is `unit_ids[page - start_page]`. The slot is that unit's
/// first token slot plus the position's offset in its page — the row of the
/// cache planes the KV-write kernel stores to.
///
/// `None` when a position lies outside the table (before `start_page` or past
/// the last unit it covers). A batch must not write a slot it does not hold, so
/// the caller treats that as a planning fault rather than clamping.
pub(crate) fn slot_mapping(table: &ipc::BlockTable, positions: Range<u32>) -> Option<Vec<i64>> {
    let page_size = u64::from(page_size(table)?);
    let start_page = u64::from(table.start_page);
    let units = u64::try_from(table.unit_ids.len()).ok()?;
    let mut slots = Vec::with_capacity(positions.len());
    for position in positions {
        let position = u64::from(position);
        let page = position / page_size;
        let unit = units.checked_sub(page.checked_sub(start_page)?)?;
        if unit == 0 {
            return None;
        }
        let index = usize::try_from(page - start_page).ok()?;
        let unit = u64::from(table.unit_ids.get(index)?.0);
        let slot = unit
            .checked_mul(page_size)?
            .checked_add(position % page_size)?;
        slots.push(i64::try_from(slot).ok()?);
    }
    Some(slots)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table of three pages of four tokens, covering pages 2..5, whose units
    /// are 7, 9 and 10 (not contiguous: the pool assigns what it has).
    fn table() -> ipc::BlockTable {
        ipc::BlockTable {
            request_pool_idx: 1,
            group_id: 0,
            start_page: 2,
            unit_ids: vec![ipc::UnitId(7), ipc::UnitId(9), ipc::UnitId(10)],
            allocated_tokens: 12,
        }
    }

    #[test]
    fn a_page_holds_the_tokens_its_units_cover() {
        assert_eq!(page_size(&table()), Some(4));

        let no_units = ipc::BlockTable {
            unit_ids: Vec::new(),
            ..table()
        };
        assert_eq!(page_size(&no_units), None, "a table without units");

        let uneven = ipc::BlockTable {
            allocated_tokens: 13,
            ..table()
        };
        assert_eq!(page_size(&uneven), None, "12 tokens over 3 units is 4 each");
    }

    #[test]
    fn a_token_writes_the_row_of_its_page_and_its_offset() {
        // Pages 2..5 hold positions 8..20, so position 8 is the first token of
        // unit 7 and position 11 the last; position 12 opens unit 9.
        assert_eq!(
            slot_mapping(&table(), 8..14).unwrap(),
            vec![7 * 4, 7 * 4 + 1, 7 * 4 + 2, 7 * 4 + 3, 9 * 4, 9 * 4 + 1]
        );
        assert_eq!(
            slot_mapping(&table(), 19..20).unwrap(),
            vec![10 * 4 + 3],
            "the last token the table covers"
        );
        assert!(slot_mapping(&table(), 8..8).unwrap().is_empty());
    }

    #[test]
    fn a_position_outside_the_table_has_no_slot() {
        // Before the table's first page, and past its last unit.
        assert_eq!(slot_mapping(&table(), 7..9), None);
        assert_eq!(slot_mapping(&table(), 19..21), None);
    }

    #[test]
    fn a_table_without_units_maps_nothing() {
        let empty = ipc::BlockTable {
            unit_ids: Vec::new(),
            ..table()
        };
        assert_eq!(slot_mapping(&empty, 8..9), None);
    }
}
