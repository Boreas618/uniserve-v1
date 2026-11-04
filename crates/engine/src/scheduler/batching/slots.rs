//! The KV slot each query token of a call writes (design §9: the mechanism
//! writes the batch in the layout the kernels read).
//!
//! A request's KV lives in physical cache units. One logical page of a cache
//! group holds `block_size` tokens and occupies `units_per_page` units
//! (`ipc::CacheGroup`, reported by the worker at handshake and kept in
//! `WorkerInfo`). The KV-write kernels address one row of the cache planes per
//! token and store it themselves as `unit * block_size + position % block_size`
//! (`ops/kernels/gather_decode_inputs`, `gather_request_rows`), so the mechanism
//! states the same number in the batch's slot column. Nothing else derives it:
//! the executor submits the column, it does not build it.
//!
//! The mapping is a pure function of a table, the group's page geometry and the
//! positions of the tokens a call evaluates, which is why it lives here rather
//! than in the call-building code.

use std::ops::Range;

/// A cache group's page shape, as the worker reported it (`ipc::CacheGroup`).
///
/// Only `block_size` is needed to turn a position into a slot; `units_per_page`
/// is carried so the mapping can refuse a page shape it cannot address (see
/// [`slot_mapping`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PageGeometry {
    /// Tokens one logical page holds.
    pub block_size: u32,
    /// Physical units one logical page occupies.
    pub units_per_page: u32,
}

/// The slot each position of `positions` writes through `table`.
///
/// A position is absolute in its request's sequence. The page holding it is
/// `position / block_size`; the table's `start_page` is the first page its
/// `unit_ids` cover, so the unit is `unit_ids[page - start_page]` and the slot
/// is that unit's first token slot plus the position's offset in its page.
///
/// `None` when:
/// - a position lies before `start_page` or past the table's last page, which is
///   a planning fault rather than something to clamp: a batch must not write a
///   slot it does not hold;
/// - the page geometry is unusable — `block_size == 0`, or `units_per_page != 1`.
///   A page of several units holds different layers' columns in each unit, so
///   one slot number per token addresses one of them; the kernels index one unit
///   per page today, and a multi-unit page needs a column per unit, which is a
///   decision for the executor rather than a guess here.
pub(crate) fn slot_mapping(
    table: &ipc::BlockTable,
    page: PageGeometry,
    positions: Range<u32>,
) -> Option<Vec<i64>> {
    if page.block_size == 0 || page.units_per_page != 1 {
        return None;
    }
    let block_size = u64::from(page.block_size);
    let start_page = u64::from(table.start_page);
    // The table's end page is the first page it does not cover: its absolute
    // token extent over the page size.
    let end_page = u64::from(table.allocated_tokens) / block_size;
    let mut slots = Vec::with_capacity(positions.len());
    for position in positions {
        let position = u64::from(position);
        let page_index = position / block_size;
        if page_index < start_page || page_index >= end_page {
            return None;
        }
        let index = usize::try_from(page_index - start_page).ok()?;
        let unit = u64::from(table.unit_ids.get(index)?.0);
        let slot = unit
            .checked_mul(block_size)?
            .checked_add(position % block_size)?;
        slots.push(i64::try_from(slot).ok()?);
    }
    Some(slots)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four tokens per page, one unit per page: the geometry every token group
    /// reports today.
    fn page() -> PageGeometry {
        PageGeometry {
            block_size: 4,
            units_per_page: 1,
        }
    }

    /// A table covering pages 2..5 (positions 8..20) whose units are 7, 9 and
    /// 10 — not contiguous, because the pool assigns what it has.
    fn table() -> ipc::BlockTable {
        ipc::BlockTable {
            request_pool_idx: 1,
            group_id: 0,
            start_page: 2,
            unit_ids: vec![ipc::UnitId(7), ipc::UnitId(9), ipc::UnitId(10)],
            allocated_tokens: 20,
        }
    }

    #[test]
    fn a_token_writes_the_row_of_its_page_and_its_offset() {
        // Position 8 is page 2's first token, so unit 7's first slot; position
        // 11 is its last; position 12 opens unit 9.
        assert_eq!(
            slot_mapping(&table(), page(), 8..14).unwrap(),
            vec![7 * 4, 7 * 4 + 1, 7 * 4 + 2, 7 * 4 + 3, 9 * 4, 9 * 4 + 1]
        );
        assert_eq!(
            slot_mapping(&table(), page(), 19..20).unwrap(),
            vec![10 * 4 + 3],
            "the last token the table covers"
        );
        assert!(slot_mapping(&table(), page(), 8..8).unwrap().is_empty());
    }

    #[test]
    fn a_position_outside_the_table_has_no_slot() {
        // Before the table's first page, and past its last.
        assert_eq!(slot_mapping(&table(), page(), 7..9), None);
        assert_eq!(slot_mapping(&table(), page(), 19..21), None);
    }

    #[test]
    fn a_page_shape_the_kernels_cannot_address_has_no_slot() {
        let multi_unit = PageGeometry {
            block_size: 4,
            units_per_page: 2,
        };
        assert_eq!(slot_mapping(&table(), multi_unit, 8..9), None);
        let no_tokens = PageGeometry {
            block_size: 0,
            units_per_page: 1,
        };
        assert_eq!(slot_mapping(&table(), no_tokens, 8..9), None);
        let empty = ipc::BlockTable {
            unit_ids: Vec::new(),
            ..table()
        };
        assert_eq!(slot_mapping(&empty, page(), 8..9), None);
    }
}
